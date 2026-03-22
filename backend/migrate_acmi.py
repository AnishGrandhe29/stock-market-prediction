import asyncio
from app.core.database import engine, Base
from app.models.prediction import Prediction

async def migrate_db():
    print("Connecting to database...")
    async with engine.begin() as conn:
        print("Dropping old predictions table...")
        await conn.run_sync(Base.metadata.drop_all, tables=[Prediction.__table__])
        print("Creating new predictions table...")
        await conn.run_sync(Base.metadata.create_all, tables=[Prediction.__table__])
    print("Migration complete!")

if __name__ == "__main__":
    asyncio.run(migrate_db())
