"""
auto_nifty_full_auto_fixed.py

Fully automated NIFTY-50 historical builder.
Replacements made:
  - Replaced broken 'nsepy' with 'yfinance' (Yahoo Finance).
  - Yahoo Finance requires '.NS' suffix for Indian stocks.
  - Fixed threading/attribute errors.

1) Fetches constituents (NSE JSON -> Wikipedia fallback -> Hardcoded fallback)
2) Downloads OHLCV via yfinance
3) Saves CSV/Excel
"""

import time
import random
from datetime import date, datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf  # REPLACEMENT FOR NSEPY
from tqdm import tqdm

# ---------- CONFIG ----------
YEARS = 10
TODAY = date.today()
DEFAULT_START = date(TODAY.year - YEARS, TODAY.month, TODAY.day)
DEFAULT_END = TODAY
POLITE_DELAY = 0.5        # Faster than nsepy, but still polite
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
OUT_CONS = Path("constituents_history_auto.csv")
OUT_CSV = Path("NIFTY50_10Y_Daily.csv")
OUT_XLSX = Path("NIFTY50_10Y_Daily.xlsx")

# ---------- Utilities ----------
def sleep_backoff(attempt=1, base=1.0):
    time.sleep(base * (RETRY_BACKOFF ** (attempt-1)) + random.random()*0.5)

# ---------- Step A: get current constituents from NSE (session + cookies) ----------
def fetch_nse_current():
    """
    Try official NSE endpoints. Returns list of uppercase symbols (without .NS).
    """
    print("Attempting to fetch constituents from NSE website...")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    
    try:
        # Prime cookies
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass

    for attempt in range(1, MAX_RETRIES+1):
        try:
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                raise RuntimeError(f"Status {r.status_code}")
            
            j = r.json()
            syms = []
            
            # Parse Strategy 1: 'data' key
            if "data" in j:
                for row in j["data"]:
                    # Grab symbol, ignore "NIFTY 50" entry if present
                    s = row.get("symbol")
                    if s and s != "NIFTY 50":
                        syms.append(s.upper())
            
            # Parse Strategy 2: Recurse (in case structure changes)
            if not syms:
                def walk(o):
                    if isinstance(o, dict):
                        for k,v in o.items():
                            if k == "symbol" and isinstance(v, str) and v != "NIFTY 50":
                                syms.append(v.upper())
                            else:
                                walk(v)
                    elif isinstance(o, list):
                        for item in o:
                            walk(item)
                walk(j)

            # Clean up
            syms = sorted(list(set([s for s in syms if s.isalpha() or ('-' in s) or ('&' in s)])))
            
            if len(syms) >= 49: # Nifty usually has 50
                return syms
            
            sleep_backoff(attempt)
        except Exception as e:
            sleep_backoff(attempt)
            continue
    return []

# ---------- Step B: fallback to Wikipedia ----------
def fetch_wikipedia_nifty():
    print("Attempting to fetch from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        # Find the table with constituents
        tables = soup.find_all("table", {"class":"wikitable"})
        for tbl in tables:
            headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
            if any("symbol" in h for h in headers):
                syms = []
                rows = tbl.find_all("tr")[1:]
                for tr in rows:
                    tds = tr.find_all("td")
                    if len(tds) > 1:
                        # Symbol is usually the 1st or 2nd column
                        cand = tds[1].get_text(strip=True) # Try col 1
                        if not cand.isupper(): 
                            cand = tds[0].get_text(strip=True) # Try col 0
                        
                        # Cleanup ".NS" if wiki included it
                        cand = cand.replace(".NS","").strip()
                        if len(cand) > 0:
                            syms.append(cand)
                if len(syms) >= 30:
                    return sorted(list(set(syms)))
    except Exception:
        pass
    return []

# ---------- Step C: write constituents CSV ----------
def write_constituents_csv(symbols, start_date=DEFAULT_START, end_date=DEFAULT_END, path=OUT_CONS):
    rows = []
    for s in sorted(symbols):
        rows.append({"Symbol": s, "Company": s, "StartDate": start_date.isoformat(), "EndDate": ""})
    df = pd.DataFrame(rows, columns=["Symbol","Company","StartDate","EndDate"])
    df.to_csv(path, index=False)
    return df

# ---------- Step D: download OHLCV via yfinance ----------
def download_for_symbols(constituents_csv=OUT_CONS, out_csv=OUT_CSV, out_xlsx=OUT_XLSX):
    ch = pd.read_csv(constituents_csv)
    # Clean column names
    ch.columns = [c.strip() for c in ch.columns]
    
    if "Symbol" not in ch.columns:
        raise ValueError("CSV missing 'Symbol' column")

    all_data = []
    failures = []
    
    # Convert inputs to list of dicts for iteration
    records = ch.to_dict('records')

    print(f"Downloading data for {len(records)} symbols via yfinance...")

    for row in tqdm(records):
        base_sym = str(row["Symbol"]).strip().upper()
        # Yahoo Finance requires .NS for NSE stocks
        yf_sym = f"{base_sym}.NS"
        company_name = row.get("Company", base_sym)
        
        # Handle Dates
        try:
            s_str = row.get("StartDate")
            start_d = s_str if pd.notna(s_str) else DEFAULT_START.isoformat()
        except:
            start_d = DEFAULT_START.isoformat()

        # Download
        try:
            # yfinance download
            # auto_adjust=True gets clean OHLC. 
            df = yf.download(yf_sym, start=start_d, end=DEFAULT_END, progress=False, auto_adjust=False)
            
            if df.empty:
                failures.append(base_sym)
                continue
                
            # Clean DataFrame
            df = df.reset_index()
            
            # Flatten MultiIndex columns if they exist (common in new yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)  # Drop the 'Ticker' level

            # Ensure columns exist and rename if necessary
            # yf typically returns: Date, Open, High, Low, Close, Adj Close, Volume
            cols_map = {
                "Date": "Date", 
                "Open": "Open", 
                "High": "High", 
                "Low": "Low", 
                "Close": "Close", 
                "Volume": "Volume"
            }
            
            # Filter only needed columns
            available_cols = [c for c in df.columns if c in cols_map]
            df = df[available_cols]
            
            # Add metadata
            df["Symbol"] = base_sym
            df["Company"] = company_name
            
            # Reorder
            final_cols = ["Symbol", "Company", "Date", "Open", "High", "Low", "Close", "Volume"]
            # Ensure all cols exist (fill missing with 0/NA if needed, though unlikely)
            for c in final_cols:
                if c not in df.columns:
                    df[c] = None
            
            df = df[final_cols]
            
            # Remove rows with NaN in prices
            df = df.dropna(subset=["Close"])
            
            all_data.append(df)

        except Exception as e:
            # print(f"Error {base_sym}: {e}")
            failures.append(base_sym)

    if not all_data:
        raise RuntimeError("No data downloaded. Check internet or yfinance version.")

    print("combining data...")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Format Date to remove timestamp if present
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.date
    
    # Sort
    combined = combined.sort_values(["Symbol", "Date"])
    
    # Save
    combined.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")
    
    # Excel can fail on huge datasets, try/except block
    try:
        combined.to_excel(out_xlsx, index=False)
        print(f"Saved Excel: {out_xlsx}")
    except Exception as e:
        print(f"Could not save Excel (data might be too large): {e}")

    if failures:
        print(f"Failed symbols ({len(failures)}): {failures}")
    
    return combined

# ---------- MAIN ----------
def main():
    print("--- Auto NIFTY-50 Builder (yfinance edition) ---")
    
    # 1. Discovery
    syms = fetch_nse_current()
    if not syms:
        print("NSE website blocked/failed. Trying Wikipedia...")
        syms = fetch_wikipedia_nifty()
    
    if not syms:
        print("Auto-discovery failed. Using Hardcoded Fallback.")
        # Top 50 approx list
        syms = ["RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","ITC","BHARTIARTL","SBIN","HINDUNILVR","LICI",
                "LT","BAJFINANCE","HCLTECH","MARUTI","SUNPHARMA","ADANIENT","TITAN","TATAMOTORS","KOTAKBANK",
                "ONGC","AXISBANK","NTPC","ULTRACEMCO","ADANIPORTS","POWERGRID","M&M","WIPRO","BAJAJFINSV",
                "COALINDIA","TATASTEEL","ASIANPAINT","JSWSTEEL","SIEMENS","TRENT","NESTLEIND","GRASIM",
                "SBILIFE","TECHM","BEL","HINDALCO","JIOFIN","TELCO","LTIM","DLF","VBL","CHOLAFIN",
                "ADANIPOWER","DRREDDY","BRITANNIA","CIPLA"]
    
    print(f"Targeting {len(syms)} symbols.")
    
    # 2. Write List
    write_constituents_csv(syms)
    
    # 3. Download
    download_for_symbols()
    print("Done.")

if __name__ == "__main__":
    main()