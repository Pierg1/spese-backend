from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import re
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Spese API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def verify_token(x_api_key: str = Header(...)):
    expected = os.environ.get("API_KEY", "")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key


# ============================================================
# NORMALIZZAZIONE OPERAZIONI (per dedup movimenti banca)
# ============================================================

ITALIAN_CITIES = {
    "vigonza", "padova", "padov", "cadoneghe", "noventa", "massanzago",
    "treviso", "trevisa", "venezia", "milano", "torino", "napoli",
    "bologna", "verona", "vicenza", "rovigo", "mestre", "firenze",
    "genova", "trieste", "bari", "palermo", "catania",
    "roma", "rome", "cork", "luxembourg", "ireland", "francisco",
    "san", "limited", "italy", "italia",
}

CORP_SUFFIXES = {
    "srl", "srls", "spa", "snc", "sas", "ltd", "inc", "gmbh",
    "scarl", "sca", "scs", "ag", "bv", "nv", "plc", "llc", "co",
    "et", "cie", "the",
}

# Pattern per identificare movimenti PayPal nelle descrizioni della banca.
# Match case-insensitive su varianti come PAYPAL, PAY PAL, PAYPAL*, ecc.
PAYPAL_PATTERN = re.compile(r"pay\s*pal", re.IGNORECASE)


def is_paypal_movement(operazione: str, dettagli: str = "") -> bool:
    """Controlla se un movimento bancario è un addebito PayPal da skippare."""
    text = f"{operazione} {dettagli}"
    return bool(PAYPAL_PATTERN.search(text))


def normalize_op(op: str) -> str:
    """Normalizza l'operazione per il confronto di dedup."""
    if not op:
        return ""
    s = op.lower().strip()
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    m = re.search(r"mandato\s+([a-z0-9]+)", s)
    if m:
        return f"mandato {m.group(1)}"

    s = re.sub(r"^(addebito|accredito)\s+diretto(\s+(disposto|ricevuto))?(\s+a\s+favore\s+di)?\s+", "", s)
    s = re.sub(r"^bonifico(\s+(disposto|ricevuto))?(\s+a\s+favore\s+di)?\s+", "", s)
    s = re.sub(r"^pagamento\s+", "", s)
    s = re.sub(r"\beffettuato il\b.*", "", s)
    s = re.sub(r"\bmediante la cart\w*\b.*", "", s)
    s = re.sub(r"\balle ore\b.*", "", s)

    s = re.sub(r"\b([a-z])\.([a-z])\.([a-z])\.?\b", r"\1\2\3", s)
    s = re.sub(r"\b([a-z])\.([a-z])\.?\b", r"\1\2", s)

    s = re.sub(r"[().,*\-_/:;]", " ", s)
    s = s.replace(".", " ")
    s = re.sub(r"\b\w*\d\w*\b", "", s)

    keep = []
    for t in s.split():
        if len(t) <= 1:
            continue
        if t in ITALIAN_CITIES or t in CORP_SUFFIXES:
            continue
        keep.append(t)

    return " ".join(keep)[:40]


# ============================================================
# PARSER FILE BANCA
# ============================================================

def parse_isybank_excel(file_bytes: bytes) -> pd.DataFrame:
    """Parsa Excel Isybank. NB: i movimenti PayPal vengono filtrati a monte
    perché PayPal è una fonte autorevole separata."""
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Lista Operazione",
        header=13
    )
    df = df.iloc[:, :8]
    df.columns = ["data", "operazione", "dettagli", "conto",
                  "contabilizzazione", "categoria", "valuta", "importo"]
    df = df.dropna(subset=["data"])
    df["data"] = pd.to_datetime(df["data"]).dt.date
    df["importo"] = pd.to_numeric(df["importo"], errors="coerce")
    df = df.dropna(subset=["importo"])
    df["dettagli"] = df["dettagli"].fillna("").astype(str)
    df["operazione"] = df["operazione"].fillna("").astype(str)
    df["categoria"] = df["categoria"].fillna("").astype(str)

    # Filtra OUT i movimenti PayPal: li gestisce il file PayPal diretto
    mask_paypal = df.apply(
        lambda r: is_paypal_movement(r["operazione"], r["dettagli"]),
        axis=1,
    )
    df_paypal_skipped = int(mask_paypal.sum())
    df = df[~mask_paypal].copy()

    # Dedup intra-file
    df["_norm"] = df["operazione"].apply(normalize_op)
    df = df.drop_duplicates(subset=["data", "_norm", "importo"])
    df = df.drop(columns=["_norm"])

    # Attacco metadati come attributi
    df.attrs["paypal_skipped"] = df_paypal_skipped
    return df


# ============================================================
# PARSER FILE PAYPAL
# ============================================================

# Descrizioni PayPal da IMPORTARE (spese reali + rimborsi)
PAYPAL_DESC_IMPORT = {
    "Pagamento Express Checkout",
    "Pagamento preautorizzato utenza",
    "Pagamento da cellulare",
    "Pagamento generico",
    "Pagamento su sito web",
    "Pagamento con credito acquirenti PayPal",
    "Trasferimento avviato dall'utente",
    # "Conversione di valuta generica" in EUR è l'addebito reale per acquisti
    # in valuta estera (USD/GBP/...). Le righe USD originali vengono ignorate
    # perché duplicherebbero l'importo.
    "Conversione di valuta generica",
    # Rimborsi (importo positivo che riduce le spese)
    "Storno pagamento",
    "Storno di versamento con addebito diretto",
    "Recupero di denaro da saldo con addebito diretto",
    "Rimborso di pagamento",
}


def parse_paypal_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parsa CSV PayPal Activity Statement (formato italiano).

    Strategia:
    - Solo righe in EUR (le valute estere sono coperte dalle righe
      "Conversione di valuta generica" che PayPal genera automaticamente).
    - Solo descrizioni che identificano spese reali o rimborsi
      (vedi PAYPAL_DESC_IMPORT). Vengono saltate le righe tecniche:
      bonifici dalla banca al wallet, blocchi/storni di preautorizzazione, ecc.
    - Per le righe "Conversione di valuta generica" il merchant non c'è,
      lo recuperiamo seguendo il "Codice transazione di riferimento" che punta
      alla riga padre (es. la riga USD del pagamento GitHub).
    - Dedup sul "Codice transazione" PayPal (univoco per definizione).
    """
    content = file_bytes.decode("utf-8-sig")
    df_full = pd.read_csv(io.StringIO(content), sep=",", quotechar='"')
    df_full.columns = df_full.columns.str.strip()

    # Mappa COMPLETA codice -> nome merchant (include anche righe USD scartate)
    cod_to_nome = dict(zip(
        df_full["Codice transazione"].astype(str),
        df_full["Nome"],
    ))

    # Filtra: solo EUR e solo descrizioni rilevanti
    df = df_full[df_full["Valuta"] == "EUR"].copy()
    df = df[df["Descrizione"].isin(PAYPAL_DESC_IMPORT)].copy()

    df["data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y").dt.date
    df["importo"] = (
        df["Netto"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    def resolve_nome(row):
        """Risolve il nome merchant, seguendo il riferimento se necessario."""
        nome = row["Nome"]
        if pd.notna(nome) and str(nome).strip():
            return str(nome).strip()
        ref = row.get("Codice transazione di riferimento")
        if pd.notna(ref):
            parent = cod_to_nome.get(str(ref))
            if pd.notna(parent) and str(parent).strip():
                return str(parent).strip()
        return "PayPal"

    df["operazione"] = df.apply(resolve_nome, axis=1)
    df["dettagli"] = df["Descrizione"].astype(str)
    df["codice_paypal"] = df["Codice transazione"].fillna("").astype(str)
    # Manteniamo "oggetto" per compatibilità con infer_category_from_paypal
    df["oggetto"] = ""

    df = df[["data", "operazione", "dettagli", "importo", "codice_paypal", "oggetto"]]
    df = df[df["codice_paypal"] != ""]
    df = df.drop_duplicates(subset=["codice_paypal"])
    return df


def infer_category_from_paypal(operazione: str, oggetto: str) -> str:
    """Indovina categoria da nome merchant + descrizione oggetto."""
    text = (operazione + " " + oggetto).lower()
    if any(k in text for k in ["openai", "claude", "anthropic", "github", "adobe", "microsoft", "apple", "google", "amazon web", "aws", "netlify", "vercel", "notion", "figma", "canva", "chatgpt", "midjourney"]):
        return "Hi-tech e informatica"
    if any(k in text for k in ["netflix", "spotify", "amazon prime", "disney", "sky", "dazn", "youtube", "twitch"]):
        return "Abbonamenti streaming"
    if any(k in text for k in ["deliveroo", "just eat", "glovo", "uber eat", "ristoran", "bar ", "pizza", "sushi", "caffe"]):
        return "Ristoranti e bar"
    if any(k in text for k in ["farma", "medic", "salute", "sanit"]):
        return "Spese mediche"
    if any(k in text for k in ["amazon", "ebay", "zalando", "shein", "abbigliamento", "vestit"]):
        return "Acquisti online"
    if any(k in text for k in ["paypal", "trasferiment", "bonifico"]):
        return "Trasferimenti"
    return "Altre uscite"


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.post("/upload", dependencies=[Depends(verify_token)])
async def upload_excel(file: UploadFile = File(...)):
    """Upload Excel Isybank. I movimenti PayPal vengono saltati (li gestisce
    il file PayPal diretto). Dedup su (data, op_norm, importo)."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "File must be .xlsx or .xls")

    content = await file.read()
    try:
        df = parse_isybank_excel(content)
    except Exception as e:
        raise HTTPException(400, f"Parse error: {e}")

    paypal_skipped = df.attrs.get("paypal_skipped", 0)

    if df.empty:
        return {
            "ok": True,
            "parsed": 0,
            "inserted": 0,
            "skipped": 0,
            "paypal_skipped": paypal_skipped,
            "date_range": None,
        }

    sb = get_supabase()

    # Carica solo i movimenti banca esistenti nel range (codice_paypal IS NULL)
    date_min = str(df["data"].min())
    date_max = str(df["data"].max())
    existing_res = (
        sb.table("movimenti")
        .select("data,op_norm,importo")
        .is_("codice_paypal", "null")
        .gte("data", date_min)
        .lte("data", date_max)
        .execute()
    )
    existing_keys = {
        (r["data"], r["op_norm"], round(float(r["importo"]), 2))
        for r in (existing_res.data or [])
    }

    inserted = 0
    skipped = 0
    new_records = []
    for _, row in df.iterrows():
        op_norm = normalize_op(row["operazione"])
        key = (str(row["data"]), op_norm, round(float(row["importo"]), 2))
        if key in existing_keys:
            skipped += 1
            continue
        existing_keys.add(key)
        new_records.append({
            "data": str(row["data"]),
            "operazione": row["operazione"],
            "op_norm": op_norm,
            "dettagli": row["dettagli"],
            "categoria": row["categoria"],
            "importo": float(row["importo"]),
            "fonte": "banca",
            "codice_paypal": None,
        })
        inserted += 1

    # Insert in batch. Se per qualche motivo passa un duplicato,
    # il constraint UNIQUE a DB lo blocca comunque.
    if new_records:
        try:
            sb.table("movimenti").insert(new_records).execute()
        except Exception as e:
            # Fallback: inserisce uno per uno saltando i conflitti
            inserted_real = 0
            for rec in new_records:
                try:
                    sb.table("movimenti").insert(rec).execute()
                    inserted_real += 1
                except Exception:
                    skipped += 1
            inserted = inserted_real

    return {
        "ok": True,
        "parsed": len(df) + paypal_skipped,
        "inserted": inserted,
        "skipped": skipped,
        "paypal_skipped": paypal_skipped,
        "date_range": {
            "from": date_min,
            "to": date_max,
        }
    }


@app.post("/upload/paypal", dependencies=[Depends(verify_token)])
async def upload_paypal(file: UploadFile = File(...)):
    """Upload CSV PayPal. Inserisce ogni transazione come movimento autonomo,
    deduplicando sul Codice transazione PayPal (univoco per definizione)."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "File must be .csv")

    content = await file.read()
    try:
        df = parse_paypal_csv(content)
    except Exception as e:
        raise HTTPException(400, f"Parse error: {e}")

    if df.empty:
        return {
            "ok": True,
            "parsed": 0,
            "inserted": 0,
            "skipped": 0,
            "date_range": None,
        }

    sb = get_supabase()

    # Carica i codici PayPal già a DB per fare dedup applicativa
    existing_res = (
        sb.table("movimenti")
        .select("codice_paypal")
        .not_.is_("codice_paypal", "null")
        .execute()
    )
    existing_codici = {r["codice_paypal"] for r in (existing_res.data or [])}

    inserted = 0
    skipped = 0
    new_records = []
    for _, row in df.iterrows():
        codice = row["codice_paypal"]
        if codice in existing_codici:
            skipped += 1
            continue
        existing_codici.add(codice)
        categoria = infer_category_from_paypal(row["operazione"], row["oggetto"])
        new_records.append({
            "data": str(row["data"]),
            "operazione": row["operazione"][:100],
            "op_norm": normalize_op(row["operazione"]),
            "dettagli": row["dettagli"] or row["oggetto"] or "",
            "categoria": categoria,
            "importo": float(row["importo"]),
            "fonte": "paypal",
            "codice_paypal": codice,
        })
        inserted += 1

    if new_records:
        try:
            sb.table("movimenti").insert(new_records).execute()
        except Exception:
            inserted_real = 0
            for rec in new_records:
                try:
                    sb.table("movimenti").insert(rec).execute()
                    inserted_real += 1
                except Exception:
                    skipped += 1
            inserted = inserted_real

    return {
        "ok": True,
        "parsed": len(df),
        "inserted": inserted,
        "skipped": skipped,
        "date_range": {
            "from": str(df["data"].min()),
            "to": str(df["data"].max()),
        }
    }


@app.post("/admin/recalc-norm", dependencies=[Depends(verify_token)])
def recalc_norm():
    """Ricalcola op_norm per tutti i movimenti banca. Da lanciare dopo aver
    modificato la logica di normalize_op()."""
    sb = get_supabase()
    rows = (
        sb.table("movimenti")
        .select("id,data,importo,operazione,op_norm")
        .is_("codice_paypal", "null")
        .execute()
        .data or []
    )

    updates = 0
    new_keys = {}
    collisions = []

    for r in rows:
        new_norm = normalize_op(r["operazione"] or "")
        key = (r["data"], new_norm, round(float(r["importo"]), 2))
        if key in new_keys:
            collisions.append({
                "kept_id": new_keys[key],
                "duplicate_id": r["id"],
                "data": r["data"],
                "importo": float(r["importo"]),
                "operazione": r["operazione"],
                "op_norm": new_norm,
            })
            continue
        new_keys[key] = r["id"]
        if new_norm != (r.get("op_norm") or ""):
            sb.table("movimenti").update({"op_norm": new_norm}).eq("id", r["id"]).execute()
            updates += 1

    return {
        "ok": True,
        "scanned": len(rows),
        "updated": updates,
        "new_collisions": len(collisions),
        "collisions": collisions[:200],
    }


@app.get("/movimenti", dependencies=[Depends(verify_token)])
def get_movimenti(from_date: str = None, to_date: str = None, fonte: str = None):
    """Lista movimenti, filtrabile per data e fonte (banca/paypal)."""
    sb = get_supabase()
    q = sb.table("movimenti").select("*").order("data", desc=True)
    if from_date:
        q = q.gte("data", from_date)
    if to_date:
        q = q.lte("data", to_date)
    if fonte in ("banca", "paypal"):
        q = q.eq("fonte", fonte)
    res = q.execute()
    return {"data": res.data, "count": len(res.data)}


@app.get("/stats", dependencies=[Depends(verify_token)])
def get_stats():
    """Statistiche di sintesi per la dashboard."""
    sb = get_supabase()
    res = sb.table("movimenti").select("data,importo,categoria,fonte").execute()
    rows = res.data
    if not rows:
        return {"total": 0}

    df = pd.DataFrame(rows)
    df["importo"] = df["importo"].astype(float)
    df["mese"] = pd.to_datetime(df["data"]).dt.to_period("M").astype(str)

    entrate = df[df["importo"] > 0]["importo"].sum()
    uscite = df[df["importo"] < 0]["importo"].abs().sum()

    by_month = df.groupby("mese").agg(
        entrate=("importo", lambda x: x[x > 0].sum()),
        uscite=("importo", lambda x: x[x < 0].abs().sum()),
    ).reset_index().to_dict(orient="records")

    by_cat = df[df["importo"] < 0].groupby("categoria")["importo"].sum().abs()
    by_cat = by_cat.sort_values(ascending=False).reset_index()
    by_cat.columns = ["categoria", "totale"]

    by_fonte = df.groupby("fonte").agg(
        n=("importo", "count"),
        uscite=("importo", lambda x: x[x < 0].abs().sum()),
    ).reset_index().to_dict(orient="records")

    return {
        "totale_movimenti": len(df),
        "entrate": round(entrate, 2),
        "uscite": round(uscite, 2),
        "saldo": round(entrate - uscite, 2),
        "by_month": by_month,
        "by_categoria": by_cat.to_dict(orient="records"),
        "by_fonte": by_fonte,
    }


@app.post("/ai/analisi", dependencies=[Depends(verify_token)])
async def ai_analisi(request: dict):
    """Chiamata Claude API per analisi delle spese."""
    import httpx

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured")

    prompt = request.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "Missing prompt")

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}]
            }
        )

    if res.status_code != 200:
        raise HTTPException(500, f"Claude API error: {res.text}")

    data = res.json()
    text = data.get("content", [{}])[0].get("text", "")
    return {"text": text}
