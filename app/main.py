"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.data.mysql_client import get_mysql_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"Starting HealthFlow API v0.1.0")
    print(f"Database: {settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}")
    print(f"vLLM: {settings.VLLM_HOST}:{settings.VLLM_PORT}")

    # Initialize database tables
    mysql_client = get_mysql_client()
    try:
        mysql_client.create_tables()
        print("Database tables created/verified")
    except Exception as e:
        print(f"Database initialization warning: {e}")

    yield

    # Shutdown
    print("Shutting down HealthFlow API")
    mysql_client.close()


app = FastAPI(
    title="HealthFlow API",
    description="HealthFlow 医疗辅助系统 - Python版本",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "HealthFlow Medical Assistant API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "healthflow-python"
    }


# Import and include routers
from app.api import chat, report, metric, kg, train

app.include_router(chat.router, prefix="/api/health", tags=["Chat"])
app.include_router(report.router, prefix="/api/health", tags=["Report"])
app.include_router(metric.router, prefix="/api/health", tags=["Metric"])
app.include_router(kg.router, prefix="/api/health", tags=["Knowledge Graph"])
app.include_router(train.router, prefix="/api/health/train", tags=["Training"])
