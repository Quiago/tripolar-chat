from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.database import create_db_and_tables
from server.routers import auth, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(
    title="FactoryMind AI API",
    description="Enterprise LLM inference platform for industrial environments.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(auth.router)
