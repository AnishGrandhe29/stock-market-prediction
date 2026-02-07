# 🚀 Steps to Run the Backend

## Quick Start

### 1. Navigate to Backend Directory
```bash
cd c:\Users\grand\Desktop\4thyrproject\backend
```

### 2. Make Sure Dependencies Are Installed
```bash
pip install -r requirements.txt
```

### 3. Start Redis (Required for Caching)
**Open a new terminal** and run:
```bash
redis-server
```

Leave this running in the background.

### 4. Start the Backend Server
**In your main terminal**, run:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Watch the Logs
You should see:
```
✅ Application started with background scheduler
🚀 Background scheduler started
📅 NIFTY data: every 15 minutes
📅 Constituents: every 30 minutes
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## What to Look For

### Good Signs ✅
- `📊 Fetching from NSE...` - Using NSE as primary source
- `✅ NSE: Fetched X days` - NSE working correctly
- `📦 Using cached data` - Cache is working
- `🔄 Running scheduled fetch` - Background jobs running

### Warning Signs ⚠️
- `⚠️ Circuit breaker open` - Too many failures, waiting 5 minutes
- `⏳ Rate limiting: waiting Xs` - Applying delays (normal)
- `❌ NSE fetch failed` - NSE down, trying Yahoo

### Error Recovery 🔧
- `📊 Fetching from Yahoo Finance` - Fallback working
- `⚠️ All sources failed, using fallback data` - All APIs down, using default values

## Testing the API

### 1. Get Realtime Price
```bash
curl http://localhost:8000/api/v1/stocks/realtime/%5ENSEI
```

**Look for** `"source": "nse"` or `"source": "yahoo_fast"` in the response.

### 2. Get Historical Data
```bash
curl http://localhost:8000/api/v1/stocks/history/%5ENSEI?days=30
```

### 3. Check Health
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Still Getting 429 Errors?
**Option 1: Increase Delays**
Edit `app/services/data_ingestion.py`:
```python
YAHOO_REQUEST_DELAY = 5  # Increase from 3 to 5 seconds
```

**Option 2: Increase Cache**
```python
CACHE_TTL_REALTIME = 600  # Increase from 300 to 10 minutes
```

**Option 3: Rely More on NSE**
The system should automatically use NSE for ^NSEI, but you can disable Yahoo temporarily by setting:
```python
CIRCUIT_BREAKER_THRESHOLD = 1  # Open circuit after 1 failure
```

### Redis Not Available?
Make sure Redis is running:
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG
```

If not installed:
```bash
# Windows: Download from https://github.com/microsoftarchive/redis/releases
# Or use WSL: sudo apt install redis-server
```

### No Data Returned?
1. Wait 5 minutes (circuit breaker timeout)
2. Check logs for specific errors
3. Clear Redis cache: `redis-cli FLUSHALL`
4. Restart the backend

## What's Different Now?

### Before
- Direct Yahoo Finance calls
- No delays → Rate limited immediately
- 2-minute cache (too short)
- Crashes on API errors

### After
- NSE India primary source (more reliable for Indian markets)
- 3-second delays between Yahoo requests
- 5-minute cache for realtime data
- Circuit breaker stops hammering failed APIs
- Background scheduler pre-fetches data every 15 minutes
- Graceful fallback when all sources fail

## Monitoring

Watch the console logs to see:
- Which data source is being used
- When circuit breakers open/close
- Cache hit/miss rates
- Background job execution

The system is now **much more resilient** and should handle rate limiting gracefully!
