from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Spese API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


def parse_isybank_excel(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse Isybank Excel export.
    - Header is at row 13 (0-indexed)
    - Dedup key: (Data, Dettagli, Importo)
    """
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
    # Dedup within this file
    df = df.drop_duplicates(subset=["data", "dettagli", "importo"])
    return df


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
        # Upsert based on unique key — Supabase will ignore duplicates
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
