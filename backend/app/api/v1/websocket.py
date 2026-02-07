"""
WebSocket API for real-time updates.
Provides live price updates and prediction notifications.
"""
from typing import Dict, Set
from datetime import datetime
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "prices": set(),
            "predictions": set(),
            "alerts": set(),
        }
    
    async def connect(self, websocket: WebSocket, channel: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)
    
    def disconnect(self, websocket: WebSocket, channel: str):
        """Remove a WebSocket connection."""
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
    
    async def broadcast(self, channel: str, message: dict):
        """Broadcast message to all connections in a channel."""
        if channel not in self.active_connections:
            return
        
        disconnected = set()
        for websocket in self.active_connections[channel]:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.active_connections[channel].discard(ws)
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to a specific connection."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
        except Exception:
            pass


manager = ConnectionManager()


@router.websocket("/prices")
async def websocket_prices(websocket: WebSocket):
    """WebSocket endpoint for real-time price updates."""
    await manager.connect(websocket, "prices")
    
    try:
        # Send initial connection confirmation
        await manager.send_personal(websocket, {
            "type": "connected",
            "channel": "prices",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        while True:
            # Keep connection alive and listen for messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                # Handle ping/pong
                if data == "ping":
                    await manager.send_personal(websocket, {"type": "pong"})
                
                # Handle subscription requests
                try:
                    msg = json.loads(data)
                    if msg.get("action") == "subscribe":
                        symbols = msg.get("symbols", ["^NSEI"])
                        await manager.send_personal(websocket, {
                            "type": "subscribed",
                            "symbols": symbols,
                        })
                except json.JSONDecodeError:
                    pass
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await manager.send_personal(websocket, {"type": "heartbeat"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, "prices")


@router.websocket("/predictions")
async def websocket_predictions(websocket: WebSocket):
    """WebSocket endpoint for prediction updates."""
    await manager.connect(websocket, "predictions")
    
    try:
        await manager.send_personal(websocket, {
            "type": "connected",
            "channel": "predictions",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                if data == "ping":
                    await manager.send_personal(websocket, {"type": "pong"})
                    
            except asyncio.TimeoutError:
                await manager.send_personal(websocket, {"type": "heartbeat"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, "predictions")


@router.websocket("/alerts/{user_id}")
async def websocket_alerts(websocket: WebSocket, user_id: int):
    """WebSocket endpoint for user-specific alerts."""
    channel = f"alerts_{user_id}"
    await manager.connect(websocket, channel)
    
    try:
        await manager.send_personal(websocket, {
            "type": "connected",
            "channel": "alerts",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                if data == "ping":
                    await manager.send_personal(websocket, {"type": "pong"})
                    
            except asyncio.TimeoutError:
                await manager.send_personal(websocket, {"type": "heartbeat"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)


# Helper function to broadcast price updates (called from scheduler)
async def broadcast_price_update(symbol: str, price_data: dict):
    """Broadcast price update to all subscribed clients."""
    await manager.broadcast("prices", {
        "type": "price_update",
        "symbol": symbol,
        "data": price_data,
        "timestamp": datetime.utcnow().isoformat(),
    })


# Helper function to broadcast new prediction (called after model inference)
async def broadcast_new_prediction(prediction_data: dict):
    """Broadcast new prediction to all subscribed clients."""
    await manager.broadcast("predictions", {
        "type": "new_prediction",
        "data": prediction_data,
        "timestamp": datetime.utcnow().isoformat(),
    })


# Helper function to send user alert (called when alert triggers)
async def send_user_alert(user_id: int, alert_data: dict):
    """Send alert to specific user."""
    channel = f"alerts_{user_id}"
    await manager.broadcast(channel, {
        "type": "alert_triggered",
        "data": alert_data,
        "timestamp": datetime.utcnow().isoformat(),
    })
