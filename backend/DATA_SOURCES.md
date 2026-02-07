# Data Sources Configuration

## Overview
The application now uses a **multi-source strategy** for fetching stock market data with automatic fallback:

1. **🥇 NSE India** (Primary - for NIFTY 50 Index)
2. **🥈 Yahoo Finance** (Backup - with rate limiting)
3. **🥉 Cached/Fallback Data** (When all sources fail)

## Features Implemented

### ✅ Rate Limiting
- **Yahoo Finance**: 3-second delay between requests
- **NSE India**: 1-second delay between requests
- Exponential backoff on retries (2s, 4s, 8s)

### ✅ Circuit Breaker
- **Threshold**: 3 consecutive failures
- **Timeout**: 5 minutes (circuit opens, blocking requests)
- Automatic reset after timeout period

### ✅ Enhanced Caching
- **Realtime data**: 5 minutes (increased from 2 minutes)
- **Historical data**: 1 hour
- **Fallback data**: 30 seconds (short cache for errors)

### ✅ Background Scheduler
- **NIFTY data**: Fetched every 15 minutes
- **Top constituents**: Fetched every 30 minutes
- Pre-populates cache to reduce on-demand API calls

## Data Sources

### 1. NSE India (National Stock Exchange)
- **Library**: `nsepython`
- **Pros**: 
  - Official Indian stock market data
  - Free and reliable
  - Low rate limits
- **Cons**: 
  - Only works for NIFTY 50 index and NSE-listed stocks
  - May require headers/cookies for some endpoints
- **Usage**: Primary source for `^NSEI`

### 2. Yahoo Finance
- **Library**: `yfinance`
- **Pros**: 
  - Global stock coverage
  - Easy to use
  - Technical indicators available
- **Cons**: 
  - Rate limiting (429 errors)
  - ~15 minute delay for free tier
  - Unreliable for Indian markets
- **Usage**: Backup source when NSE fails

### 3. Fallback Data
- Returns default values when all sources fail
- Prevents API errors from breaking the application
- Default NIFTY value: 22,000

## Configuration

### Environment Variables
No new environment variables needed currently. The application uses:
- `REDIS_URL` - for caching
- `DATABASE_URL` - for storing historical data

### Optional: Add Alternative Data Sources
To add more data sources like Alpha Vantage or Twelve Data:

```bash
# .env file
ALPHA_VANTAGE_API_KEY=your_api_key_here
TWELVE_DATA_API_KEY=your_api_key_here
```

## How It Works

### Historical Data Fetching Flow
```
Request → Check Cache → NSE (if ^NSEI) → Yahoo (fallback) → DB Cache → Fallback
```

### Realtime Price Flow
```
Request → Check Cache (5min) → NSE → Yahoo (with delays) → Fallback
```

### Circuit Breaker Flow
```
API Call → Success? → Reset counter
         → Failure? → Increment counter
                   → 3 failures? → Open circuit for 5 minutes
```

## Monitoring

### Log Messages
- `📦 Using cached data` - Data served from cache
- `📊 Fetching from NSE/Yahoo` - Making API request
- `✅ Fetched X days` - Successful fetch
- `⚠️ Circuit breaker open` - Source temporarily blocked
- `❌ Fetch failed` - Error occurred
- `🔄 Circuit breaker reset` - Circuit closed after timeout

### Rate Limiting Messages
- `⏳ Rate limiting: waiting Xs` - Delay before request

## Troubleshooting

### Problem: Still getting 429 errors
**Solution**: 
- Increase `YAHOO_REQUEST_DELAY` in `data_ingestion.py`
- Increase cache TTL values
- Rely more on NSE as primary source

### Problem: No data returned
**Solution**:
- Check if circuit breaker is open (wait 5 minutes)
- Verify Redis is running for caching
- Check database connection
- Review logs for specific errors

### Problem: Data is stale
**Solution**:
- Reduce cache TTL values
- Increase scheduler frequency (not recommended due to rate limits)
- Clear Redis cache: `redis-cli FLUSHALL`

## Performance Optimization

### Current Settings
- Cache realtime: **5 minutes** ✅
- Cache historical: **1 hour** ✅
- Yahoo delay: **3 seconds** ✅
- NSE delay: **1 second** ✅
- Scheduler: **15/30 minutes** ✅

### Recommended for High Traffic
```python
# In data_ingestion.py
CACHE_TTL_REALTIME = 600  # 10 minutes
YAHOO_REQUEST_DELAY = 5   # 5 seconds
# Increase scheduler intervals to 30/60 minutes
```

### Recommended for Development
```python
# In data_ingestion.py
CACHE_TTL_REALTIME = 60   # 1 minute (fresher data)
# Disable scheduler to avoid API calls
```

## API Usage

### Get Realtime Price
```bash
GET /api/v1/stocks/realtime/%5ENSEI
```
Response includes `source` field showing data origin: `nse`, `yahoo_fast`, `yahoo_info`, or `fallback`

### Get Historical Data
```bash
GET /api/v1/stocks/history/%5ENSEI?days=60
```
Uses database cache first, fetches from API if needed

### Force Refresh
```bash
POST /api/v1/stocks/refresh/%5ENSEI
```
Clears cache and fetches fresh data

## Migration Notes

### From Old Version
The old version relied solely on Yahoo Finance. The new version:
1. **Auto-migrates**: No code changes needed in API calls
2. **Better reliability**: NSE as primary reduces 429 errors
3. **Auto-caching**: Background scheduler pre-fetches data
4. **Graceful degradation**: Fallback data when sources fail

### Next Steps (Optional)
1. Add Alpha Vantage as third-tier backup
2. Implement CSV data loading for development
3. Add WebSocket support for real-time updates
4. Implement data quality monitoring
