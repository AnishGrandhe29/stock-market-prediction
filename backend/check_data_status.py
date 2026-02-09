import asyncio
import sys
from pathlib import Path
from app.config import settings

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("DIAGNOSTIC CHECK - MODEL ONLY")
    
    # 1. Check Model Path
    model_path = Path(settings.model_path)
    print(f"Configured: {settings.model_path}")
    print(f"Resolved:   {model_path.resolve()}")
    if model_path.exists():
        print("[OK] Model file FOUND.")
    else:
        print("[MISSING] Model file NOT FOUND.")

if __name__ == "__main__":
    main()
