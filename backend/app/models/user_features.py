"""
User feature models: Notes, Watchlist, Alerts.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


class Note(Base):
    """User notes attached to stocks or predictions."""
    
    __tablename__ = "notes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Note content
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    
    # Optional attachment to stock/prediction
    symbol = Column(String(20), nullable=True)
    prediction_id = Column(Integer, nullable=True)
    
    # Tags (comma-separated)
    tags = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="notes")
    
    def __repr__(self):
        return f"<Note {self.id} by User {self.user_id}>"


class WatchlistItem(Base):
    """User's watchlist of stocks."""
    
    __tablename__ = "watchlist"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    
    # Display name
    display_name = Column(String(100), nullable=True)
    
    # Order in watchlist
    sort_order = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="watchlist")
    
    def __repr__(self):
        return f"<WatchlistItem {self.symbol} for User {self.user_id}>"


class Alert(Base):
    """Price alerts for users."""
    
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    
    # Alert conditions
    alert_type = Column(String(50), nullable=False)  # price_above, price_below, change_pct
    target_value = Column(Float, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_triggered = Column(Boolean, default=False)
    triggered_at = Column(DateTime, nullable=True)
    triggered_value = Column(Float, nullable=True)
    
    # Notification settings
    notify_browser = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="alerts")
    
    def __repr__(self):
        return f"<Alert {self.alert_type} {self.target_value} for {self.symbol}>"
