"""
FastAPI application entry point.
Sets up routes, middleware, and application lifecycle events.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.config import settings
from app.core.database import init_db, close_db
from app.api.v1 import auth, stocks, predictions, users, websocket, news, model_info
from app.services.scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    await init_db()
    start_scheduler()  # Start background data fetching
    print("✅ Application started with background scheduler")
    yield
    # Shutdown
    stop_scheduler()  # Stop background jobs
    await close_db()
    print("👋 Application shutdown complete")


app = FastAPI(
    title="NIFTY 50 Prediction API",
    description="Multimodal deep learning predictions for NIFTY 50 Index with Explainable AI",
    version="1.0.0",
    lifespan=lifespan,
)

# Session middleware (required for OAuth)
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(stocks.router, prefix="/api/v1/stocks", tags=["Stocks"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(websocket.router, prefix="/api/v1/ws", tags=["WebSocket"])
app.include_router(news.router, prefix="/api/v1/news", tags=["News"])
app.include_router(model_info.router, prefix="/api/v1/model", tags=["Model"])


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "NIFTY 50 Prediction API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "connected",
        "model": "loaded"
    }
