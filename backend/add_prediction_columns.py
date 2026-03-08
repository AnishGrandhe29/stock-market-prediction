"""
Add trend, signal, and confidence_score columns to predictions table.
Safe to run multiple times — checks if columns exist first.
"""
import sqlite3
import sys
from pathlib import Path

# The backend's working directory is backend/, so the DB is there
DB_PATH = Path(__file__).parent / "nifty50_predict.db"

COLUMNS_TO_ADD = [
    ("trend", "TEXT"),
    ("signal", "TEXT"),
    ("confidence_score", "REAL"),
]


def main():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Check if predictions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    if not cursor.fetchone():
        print("  ⚠ predictions table does not exist yet. It will be created on next server start.")
        conn.close()
        return

    # Get existing columns
    cursor.execute("PRAGMA table_info(predictions)")
    existing = {row[1] for row in cursor.fetchall()}

    added = 0
    for col_name, col_type in COLUMNS_TO_ADD:
        if col_name in existing:
            print(f"  ✓ Column '{col_name}' already exists — skipping")
        else:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"  + Added column '{col_name}' ({col_type})")
            added += 1

    conn.commit()
    conn.close()

    if added:
        print(f"\n✅ Migration complete — {added} column(s) added.")
    else:
        print("\n✅ All columns already exist — nothing to do.")


if __name__ == "__main__":
    print("=" * 50)
    print("📦 Migrating predictions table")
    print("=" * 50)
    main()
