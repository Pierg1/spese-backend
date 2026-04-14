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


def normalize_op(op: str) -> str:
    """Normalize operation string for dedup comparison."""
    op = op.lower().strip()
    op = re.sub(r'\b(effettuato il|mediante la cart\w*|alle ore|san francisco|ireland|limited|s\.?r\.?l\.?|s\.?p\.?a\.?)\b', '', op)
    op = re.sub(r'\d{6,}', '#', op)
    op = re.sub(r'\s+', ' ', op).strip()
    return op[:80]


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
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        record = {
            "data": str(row["data"]),
            "operazione": row["operazione"],
            "dettagli": row["dettagli"],
            "categoria": row["categoria"],
            "importo": float(row["importo"]),
        }
        res = sb.table("movimenti").upsert(
            record,
            on_conflict="data,dettagli,importo"
        ).execute()
        if res.data:
            inserted += 1
        else:
            skipped += 1

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
    For each PayPal transaction, try to enrich category by matching
    existing Isybank 'PayPal' movements by date+amount.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "File must be .csv")

    content = await file.read()
    try:
        df = parse_paypal_csv(content)
    except Exception as e:
        raise HTTPException(400, f"Parse error: {e}")

    sb = get_supabase()

    # Load existing Isybank PayPal movements for enrichment matching
    existing_res = sb.table("movimenti").select("data,importo,categoria,operazione").ilike("operazione", "%paypal%").execute()
    existing_paypal = existing_res.data or []

    # Build lookup: (date_str, amount) -> categoria
    paypal_lookup = {}
    for m in existing_paypal:
        key = (m["data"], round(float(m["importo"]), 2))
        if m.get("categoria") and m["categoria"] not in ("", "Altre uscite", "Domiciliazioni e Utenze"):
            paypal_lookup[key] = m["categoria"]

    inserted = 0
    skipped = 0
    enriched = 0

    for _, row in df.iterrows():
        data_str = str(row["data"])
        importo = round(float(row["importo"]), 2)

        # Try to match with existing Isybank PayPal movement
        # Check same day and ±1 day window
        categoria = None
        for delta in [0, -1, 1]:
            from datetime import date
            check_date = (row["data"] + timedelta(days=delta)).isoformat()
            match_key = (check_date, importo)
            if match_key in paypal_lookup:
                categoria = paypal_lookup[match_key]
                enriched += 1
                break

        if not categoria:
            categoria = infer_category_from_paypal(row["operazione"], row["oggetto"])

        dettagli = f"PayPal - {row['codice']}"
        record = {
            "data": data_str,
            "operazione": row["operazione"],
            "dettagli": dettagli,
            "categoria": categoria,
            "importo": importo,
        }

        res = sb.table("movimenti").upsert(
            record,
            on_conflict="data,dettagli,importo"
        ).execute()
        if res.data:
            inserted += 1
        else:
            skipped += 1

    return {
        "ok": True,
        "parsed": len(df),
        "inserted": inserted,
        "skipped": skipped,
        "enriched": enriched,
        "date_range": {
            "from": str(df["data"].min()),
            "to": str(df["data"].max()),
        }
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
