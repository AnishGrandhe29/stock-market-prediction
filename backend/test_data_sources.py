"""
Quick test script to verify multi-source data fetching.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.data_ingestion import (
    fetch_from_nse,
    fetch_from_yahoo,
    get_realtime_price
)


async def test_data_sources():
    print("=" * 60)
    print("Testing Multi-Source Data Fetching")
    print("=" * 60)
    
    # Test 1: NSE Data Fetch
    print("\n1. Testing NSE India data fetch...")
    nse_df = await fetch_from_nse("^NSEI", days=30)
    if nse_df is not None and not nse_df.empty:
        print(f"   SUCCESS: Got {len(nse_df)} days of NSE data")
        print(f"   Latest: {nse_df.iloc[-1]['Close']:.2f}")
    else:
        print("   SKIPPED/FAILED: NSE data not available")
    
    # Test 2: Yahoo Finance Fetch (with delay to avoid rate limit)
    print("\n2. Testing Yahoo Finance data fetch...")
    await asyncio.sleep(3)  # Rate limiting delay
    yahoo_df = await fetch_from_yahoo("^NSEI", days=30)
    if yahoo_df is not None and not yahoo_df.empty:
        print(f"   SUCCESS: Got {len(yahoo_df)} days of Yahoo data")
        print(f"   Latest: {yahoo_df.iloc[-1]['Close']:.2f}")
    else:
        print("   FAILED: Yahoo data not available")
    
    # Test 3: Realtime Price (uses multi-source)
    print("\n3. Testing realtime price with fallback...")
    await asyncio.sleep(3)  # Rate limiting delay
    price_data = await get_realtime_price("^NSEI")
    print(f"   Price: {price_data['price']:.2f}")
    print(f"   Source: {price_data.get('source', 'unknown')}")
    print(f"   Change: {price_data['change_pct']:.2f}%")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_data_sources())
