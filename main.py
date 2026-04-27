from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import re
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Spese API", version="1.0.0")

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


def normalize_op(op: str) -> str:
    """Normalize operation string for dedup comparison.

    Strategy:
    - For SDD/PayPal/wire payments, the MANDATO id is the strongest fingerprint
      and is used alone. All variants ("PayPal Europe S.a.r.l." vs
      "PayPal (Europe) S.a r.l.") collapse to the same key.
    - For card payments at merchants, strip city names, corporate suffixes,
      transaction codes and markdown link artifacts, then keep the meaningful
      tokens. Truncated to 40 chars.
    """
    if not op:
        return ""
    s = op.lower().strip()

    # Markdown link artifact: "[text](url)" -> "text"
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # Direct debit / SDD: keep the mandate id only — it uniquely identifies the
    # creditor, regardless of how the bank formatted the description.
    m = re.search(r"mandato\s+([a-z0-9]+)", s)
    if m:
        return f"mandato {m.group(1)}"

    # Strip verbose Italian banking prefixes
    s = re.sub(r"^(addebito|accredito)\s+diretto(\s+(disposto|ricevuto))?(\s+a\s+favore\s+di)?\s+", "", s)
    s = re.sub(r"^bonifico(\s+(disposto|ricevuto))?(\s+a\s+favore\s+di)?\s+", "", s)
    s = re.sub(r"^pagamento\s+", "", s)
    s = re.sub(r"\beffettuato il\b.*", "", s)
    s = re.sub(r"\bmediante la cart\w*\b.*", "", s)
    s = re.sub(r"\balle ore\b.*", "", s)

    # Collapse dotted acronyms ("s.r.l." -> "srl", "s.n.c." -> "snc")
    s = re.sub(r"\b([a-z])\.([a-z])\.([a-z])\.?\b", r"\1\2\3", s)
    s = re.sub(r"\b([a-z])\.([a-z])\.?\b", r"\1\2", s)

    # Replace punctuation with spaces
    s = re.sub(r"[().,*\-_/:;]", " ", s)
    s = s.replace(".", " ")

    # Drop transaction codes: any token containing a digit
    s = re.sub(r"\b\w*\d\w*\b", "", s)

    # Tokenize and filter
    keep = []
    for t in s.split():
        if len(t) <= 1:
            continue
        if t in ITALIAN_CITIES or t in CORP_SUFFIXES:
            continue
        keep.append(t)

    return " ".join(keep)[:40]

def parse_isybank_excel(file_bytes: bytes) -> pd.DataFrame:
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
    # Dedup within file using normalized operation name
    df["_norm"] = df["operazione"].apply(normalize_op)
    df = df.drop_duplicates(subset=["data", "_norm", "importo"])
    df = df.drop(columns=["_norm"])
    return df


def parse_paypal_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parse PayPal CSV export (Italian format)."""
    content = file_bytes.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(content), sep=",", quotechar='"')
    df.columns = df.columns.str.strip()
    # Keep only completed EUR debits
    df = df[df["Valuta"] == "EUR"]
    df = df[df["Stato"] == "Completata"]
    df = df[df["Impatto sul saldo"] == "Addebito"]
    df["data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y").dt.date
    df["importo"] = df["Netto"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)
    df["operazione"] = df["Nome"].fillna("PayPal").astype(str)
    df["dettagli"] = df["Tipo"].fillna("").astype(str)
    df["codice"] = df["Codice transazione"].fillna("").astype(str)
    df["oggetto"] = df.get("Oggetto", pd.Series([""] * len(df))).fillna("").astype(str)
    df = df[["data", "operazione", "dettagli", "importo", "codice", "oggetto"]]
    df = df[df["importo"] < 0]
    df = df.drop_duplicates(subset=["codice"])
    return df


def infer_category_from_paypal(operazione: str, oggetto: str) -> str:
    """Guess category from PayPal merchant name and item description."""
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


@app.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.post("/upload", dependencies=[Depends(verify_token)])
async def upload_excel(file: UploadFile = File(...)):
    """Upload Isybank Excel. Returns inserted/skipped counts."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "File must be .xlsx or .xls")

    content = await file.read()
    try:
        df = parse_isybank_excel(content)
    except Exception as e:
        raise HTTPException(400, f"Parse error: {e}")

    sb = get_supabase()

    date_min = str(df["data"].min())
    date_max = str(df["data"].max())
    existing_res = (
        sb.table("movimenti")
        .select("data,op_norm,importo")
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
        })
        inserted += 1

    if new_records:
        sb.table("movimenti").upsert(
            new_records,
            on_conflict="data,op_norm,importo",
            ignore_duplicates=True,
        ).execute()

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


@app.post("/upload/paypal", dependencies=[Depends(verify_token)])
async def upload_paypal(file: UploadFile = File(...)):
    """
    Upload PayPal CSV.
    Instead of inserting new rows, UPDATES existing Isybank movements
    that have 'PayPal' in the description, matching by date±1 and amount.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "File must be .csv")

    content = await file.read()
    try:
        df = parse_paypal_csv(content)
    except Exception as e:
        raise HTTPException(400, f"Parse error: {e}")

    sb = get_supabase()

    # Load existing Isybank PayPal movements
    existing_res = sb.table("movimenti").select("id,data,importo,operazione,categoria").ilike("operazione", "%paypal%").execute()
    existing_paypal = existing_res.data or []

    # Build lookup: (date_str, abs_amount) -> record
    paypal_lookup = {}
    for m in existing_paypal:
        key = (m["data"], abs(round(float(m["importo"]), 2)))
        paypal_lookup[key] = m

    updated = 0
    not_found = 0

    for _, row in df.iterrows():
        importo_abs = abs(round(float(row["importo"]), 2))
        
        # Try date ±1 day
        matched = None
        for delta in [0, -1, 1]:
            from datetime import date as date_cls, timedelta
            check_date = (row["data"] + timedelta(days=delta)).isoformat()
            key = (check_date, importo_abs)
            if key in paypal_lookup:
                matched = paypal_lookup[key]
                break

        if matched:
            # Infer category from PayPal merchant name
            categoria = infer_category_from_paypal(row["operazione"], row["oggetto"])
            # Only update if current category is generic
            generic_cats = ["", "Altre uscite", "Domiciliazioni e Utenze", "Addebiti vari"]
            if matched["categoria"] in generic_cats:
                sb.table("movimenti").update({
                    "operazione": row["operazione"][:100],
                    "categoria": categoria,
                }).eq("id", matched["id"]).execute()
                updated += 1
            else:
                updated += 1  # already has good category, still count as matched
        else:
            not_found += 1

    return {
        "ok": True,
        "parsed": len(df),
        "updated": updated,
        "not_found": not_found,
        "date_range": {
            "from": str(df["data"].min()),
            "to": str(df["data"].max()),
        }
    }

@app.post("/admin/recalc-norm", dependencies=[Depends(verify_token)])
def recalc_norm():
    """Recompute op_norm for every row using the current normalize_op().
    Run once after changing the normalization logic. Returns a summary and
    a list of (data, importo) groups that became duplicates so they can be
    cleaned up manually."""
    sb = get_supabase()
    rows = sb.table("movimenti").select("id,data,importo,operazione,op_norm").execute().data or []

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
def get_movimenti(from_date: str = None, to_date: str = None):
    """Return all movimenti, optionally filtered by date range."""
    sb = get_supabase()
    q = sb.table("movimenti").select("*").order("data", desc=True)
    if from_date:
        q = q.gte("data", from_date)
    if to_date:
        q = q.lte("data", to_date)
    res = q.execute()
    return {"data": res.data, "count": len(res.data)}


@app.get("/stats", dependencies=[Depends(verify_token)])
def get_stats():
    """Summary stats for dashboard."""
    sb = get_supabase()
    res = sb.table("movimenti").select("data,importo,categoria").execute()
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

    return {
        "totale_movimenti": len(df),
        "entrate": round(entrate, 2),
        "uscite": round(uscite, 2),
        "saldo": round(entrate - uscite, 2),
        "by_month": by_month,
        "by_categoria": by_cat.to_dict(orient="records"),
    }


@app.post("/ai/analisi", dependencies=[Depends(verify_token)])
async def ai_analisi(request: dict):
    """Call Claude API to analyze spending data."""
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
