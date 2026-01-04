import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
import os

def fetch_macro_data(start_date="2015-01-01", end_date="2025-12-31"):
    """
    Fetches Macro and ESG data from FRED and Yahoo Finance.
    """
    print("Fetching Macro Data...")
    
    # 1. FRED Data
    # CPI: Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL) - Monthly
    # IIP: Industrial Production: Total Index (INDPRO) - Monthly
    # 10Y Yield: Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity (DGS10) - Daily
    # Note: Using US data as proxies for global macro if Indian specific series are hard to get via simple API without keys.
    # Ideally we would use Indian CPI/IIP but FRED is easiest for demo. 
    # Let's try to get INR specific if possible or just use US as global macro factors.
    # Actually, for NIFTY, USDINR is crucial.
    
    # FRED Series IDs
    fred_series = {
        "CPI": "CPIAUCSL", # Monthly
        "IIP": "INDPRO",   # Monthly
        "US10Y": "DGS10",  # Daily
    }
    
    try:
        fred_data = web.DataReader(list(fred_series.values()), "fred", start_date, end_date)
        fred_data = fred_data.rename(columns={v: k for k, v in fred_series.items()})
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        fred_data = pd.DataFrame()

    # 2. Yahoo Finance Data
    # USDINR=X: USD/INR Exchange Rate
    # CL=F: Crude Oil Futures
    # ^NSEI: Nifty 50 (for reference/alignment if needed)
    # ESGBEES.NS: Nifty 100 ESG Sector Leaders ETF (Proxy for ESG)
    
    yf_tickers = {
        "USDINR": "USDINR=X",
        "CrudeOil": "CL=F",
        "ESG_ETF": "ESGBEES.NS" 
    }
    
    yf_data_list = []
    for name, ticker in yf_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # Use 'Close' price
                series = df["Close"]
                if isinstance(series, pd.DataFrame):
                     series = series.iloc[:, 0] # Handle multi-index if present
                series.name = name
                yf_data_list.append(series)
        except Exception as e:
            print(f"Error fetching {name} ({ticker}): {e}")

    if yf_data_list:
        yf_df = pd.concat(yf_data_list, axis=1)
    else:
        yf_df = pd.DataFrame()

    # Combine
    macro_df = pd.concat([fred_data, yf_df], axis=1)
    
    # Resample to daily and forward fill (Macro data is often monthly)
    macro_df = macro_df.resample("D").ffill()
    
    # Filter to date range
    macro_df = macro_df[(macro_df.index >= start_date) & (macro_df.index <= end_date)]
    
    # Fill remaining NaNs (e.g. holidays)
    macro_df = macro_df.ffill().bfill()
    
    # Save
    output_path = os.path.join("Datasets", "macro_data.csv")
    os.makedirs("Datasets", exist_ok=True)
    macro_df.to_csv(output_path)
    print(f"Macro data saved to {output_path}")
    print(macro_df.head())
    print(macro_df.tail())

if __name__ == "__main__":
    fetch_macro_data()
