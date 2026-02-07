"""
User management API endpoints.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import get_current_user, get_password_hash
from app.models.user import User
from app.models.user_features import Note, WatchlistItem, Alert
from app.schemas import (
    UserResponse, UserUpdate,
    NoteCreate, NoteUpdate, NoteResponse,
    WatchlistItemCreate, WatchlistItemResponse,
    AlertCreate, AlertResponse,
)

router = APIRouter()


# ============ User Profile ============

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user profile."""
    result = await db.execute(select(User).where(User.id == int(current_user["user_id"])))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.patch("/me", response_model=UserResponse)
async def update_profile(
    update_data: UserUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user profile."""
    result = await db.execute(select(User).where(User.id == int(current_user["user_id"])))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if update_data.full_name is not None:
        user.full_name = update_data.full_name
    if update_data.avatar_url is not None:
        user.avatar_url = update_data.avatar_url
    
    await db.commit()
    await db.refresh(user)
    
    return user


# ============ Notes ============

@router.get("/notes", response_model=List[NoteResponse])
async def get_notes(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all notes for current user."""
    result = await db.execute(
        select(Note)
        .where(Note.user_id == int(current_user["user_id"]))
        .order_by(Note.updated_at.desc())
    )
    return result.scalars().all()


@router.post("/notes", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note(
    note_data: NoteCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new note."""
    note = Note(
        user_id=int(current_user["user_id"]),
        title=note_data.title,
        content=note_data.content,
        symbol=note_data.symbol,
        tags=note_data.tags,
    )
    db.add(note)
    await db.commit()
    await db.refresh(note)
    
    return note


@router.patch("/notes/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: int,
    update_data: NoteUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a note."""
    result = await db.execute(
        select(Note).where(
            Note.id == note_id,
            Note.user_id == int(current_user["user_id"])
        )
    )
    note = result.scalar_one_or_none()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    if update_data.title is not None:
        note.title = update_data.title
    if update_data.content is not None:
        note.content = update_data.content
    if update_data.tags is not None:
        note.tags = update_data.tags
    
    await db.commit()
    await db.refresh(note)
    
    return note


@router.delete("/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_note(
    note_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a note."""
    result = await db.execute(
        select(Note).where(
            Note.id == note_id,
            Note.user_id == int(current_user["user_id"])
        )
    )
    note = result.scalar_one_or_none()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    await db.delete(note)
    await db.commit()


# ============ Watchlist ============

@router.get("/watchlist", response_model=List[WatchlistItemResponse])
async def get_watchlist(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's watchlist."""
    result = await db.execute(
        select(WatchlistItem)
        .where(WatchlistItem.user_id == int(current_user["user_id"]))
        .order_by(WatchlistItem.sort_order)
    )
    return result.scalars().all()


@router.post("/watchlist", response_model=WatchlistItemResponse, status_code=status.HTTP_201_CREATED)
async def add_to_watchlist(
    item_data: WatchlistItemCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add stock to watchlist."""
    # Check if already in watchlist
    result = await db.execute(
        select(WatchlistItem).where(
            WatchlistItem.user_id == int(current_user["user_id"]),
            WatchlistItem.symbol == item_data.symbol
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Already in watchlist")
    
    item = WatchlistItem(
        user_id=int(current_user["user_id"]),
        symbol=item_data.symbol,
        display_name=item_data.display_name,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    
    return item


@router.delete("/watchlist/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_from_watchlist(
    item_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Remove stock from watchlist."""
    result = await db.execute(
        select(WatchlistItem).where(
            WatchlistItem.id == item_id,
            WatchlistItem.user_id == int(current_user["user_id"])
        )
    )
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    await db.delete(item)
    await db.commit()


# ============ Alerts ============

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's alerts."""
    result = await db.execute(
        select(Alert)
        .where(Alert.user_id == int(current_user["user_id"]))
        .order_by(Alert.created_at.desc())
    )
    return result.scalars().all()


@router.post("/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    alert_data: AlertCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a price alert."""
    alert = Alert(
        user_id=int(current_user["user_id"]),
        symbol=alert_data.symbol,
        alert_type=alert_data.alert_type,
        target_value=alert_data.target_value,
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    
    return alert


@router.delete("/alerts/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert(
    alert_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an alert."""
    result = await db.execute(
        select(Alert).where(
            Alert.id == alert_id,
            Alert.user_id == int(current_user["user_id"])
        )
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    await db.delete(alert)
    await db.commit()
