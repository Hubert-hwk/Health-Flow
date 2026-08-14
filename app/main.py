"""FastAPI entry point for HealthFlow."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.data.milvus_client import get_milvus_client
from app.data.mysql_client import get_mysql_client
from app.data.neo4j_client import get_neo4j_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database = get_mysql_client()
    try:
        database.create_tables()
        yield
    finally:
        database.close()


app = FastAPI(
    title="HealthFlow 医疗辅助系统",
    description="面向体检报告理解、证据检索和安全问答的医疗辅助系统。不能替代医生。",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "HealthFlow Medical Assistant",
        "message": "HealthFlow Medical Assistant API",
        "version": "0.1.0",
        "scope": "medical-assistance-only",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "healthflow-python",
        "database": settings.database_url.split(":", 1)[0],
    }


@app.get("/ready")
async def readiness_check():
    database = get_mysql_client()
    milvus = get_milvus_client()
    neo4j = get_neo4j_client()

    db_ok = False
    try:
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ready" if db_ok else "degraded",
        "database": "ok" if db_ok else "unavailable",
        "milvus": "ok" if _milvus_probe(milvus) else "optional_unavailable",
        "neo4j": "ok" if _neo4j_probe(neo4j) else "optional_unavailable",
    }


def _milvus_probe(client) -> bool:
    """真实的 Milvus 连通性探测：客户端对象存在不代表服务可用。"""
    try:
        return bool(client.client and client.client.list_collections() is not None)
    except Exception:
        return False


def _neo4j_probe(client) -> bool:
    """真实的 Neo4j 连通性探测：执行一次 RETURN 1。"""
    try:
        return bool(client.connect())
    except Exception:
        return False


from app.api import chat, kg, metric, report, train  # noqa: E402

app.include_router(chat.router, prefix="/api/health", tags=["Chat"])
app.include_router(report.router, prefix="/api/health", tags=["Report"])
app.include_router(metric.router, prefix="/api/health", tags=["Metric"])
app.include_router(kg.router, prefix="/api/health", tags=["Knowledge Graph"])
app.include_router(train.router, prefix="/api/health/train", tags=["Training"])
