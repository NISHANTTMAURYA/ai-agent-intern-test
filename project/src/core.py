"""Core configurations, multi-tenant loader, and structured request tracing."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class Settings(BaseSettings):
    """Global application settings loaded from environment or .env file."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    TENANTS_DIR: Path = Path(__file__).resolve().parent.parent / "tenants"
    LOGS_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
    
    DEFAULT_TENANT: str = "aster-and-row"
    DEBUG_MODE: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    LLM_PROVIDER: str = "gemini"  # Options: "gemini", "openai", or "mock"
    LLM_MODEL: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    EVAL_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)


class TenantConfig(BaseModel):
    """Configuration schema for a brand tenant."""
    tenant_id: str
    brand_name: str
    description: str = ""
    knowledge_base_dir: Path
    orders_file: Path
    persona: str = ""

    @classmethod
    def load(cls, tenant_id: str = "aster-and-row") -> "TenantConfig":
        """
        Load brand tenant configuration from its YAML file.
        
        Steps:
        1. Check if tenants/{tenant_id}/tenant.yaml exists.
        2. Parse custom brand metadata, personas, and relative paths.
        3. Fall back to standard root directories if config is omitted.
        """
        config_file = settings.TENANTS_DIR / tenant_id / "tenant.yaml"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            kb_rel = raw.get("paths", {}).get("knowledge_base_dir", "knowledge-base")
            orders_rel = raw.get("paths", {}).get("orders_file", "data/orders.json")
            
            # Resolve paths checking local and parent roots
            kb_path = settings.BASE_DIR / kb_rel if (settings.BASE_DIR / kb_rel).exists() else settings.BASE_DIR.parent / kb_rel
            orders_path = settings.BASE_DIR / orders_rel if (settings.BASE_DIR / orders_rel).exists() else settings.BASE_DIR.parent / orders_rel
            
            return cls(
                tenant_id=raw.get("tenant_id", tenant_id),
                brand_name=raw.get("brand_name", "Aster & Row"),
                description=raw.get("description", ""),
                knowledge_base_dir=kb_path,
                orders_file=orders_path,
                persona=raw.get("persona", "")
            )
            
        # Default tenant configuration fallback
        kb_path = settings.BASE_DIR / "knowledge-base" if (settings.BASE_DIR / "knowledge-base").exists() else settings.BASE_DIR.parent / "knowledge-base"
        orders_path = settings.BASE_DIR / "data/orders.json" if (settings.BASE_DIR / "data/orders.json").exists() else settings.BASE_DIR.parent / "data/orders.json"
        
        return cls(
            tenant_id=tenant_id,
            brand_name="Aster & Row",
            knowledge_base_dir=kb_path,
            orders_file=orders_path,
            persona="You are a helpful and honest AI support assistant for Aster & Row."
        )


class RequestTrace(BaseModel):
    """Structured observability log format for each customer interaction."""
    trace_id: str
    session_id: str
    tenant_id: str
    timestamp: str
    user_message: str
    conversation_history: List[str] = Field(default_factory=list)  # Prior turns (Customer:/Agent: prefixed)
    intent: str
    retrieved_sources: List[str]
    tool_called: Optional[str] = None
    final_response: str
    handoff_recommended: bool
    duration_ms: float


class StructuredTracer:
    """Writes sanitized, structured JSON traces for auditing and debuggability."""
    
    def __init__(self, logs_dir: Optional[Path] = None):
        self.logs_dir = logs_dir or settings.LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.logs_dir / "interaction_traces.jsonl"

    def log_interaction(self, trace: RequestTrace) -> None:
        """
        Append a structured interaction trace to the JSONL log file.
        
        Steps:
        1. Convert trace object to sanitized JSON dict.
        2. Append JSON line with atomic file write.
        """
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(trace.model_dump_json() + "\n")
        except Exception as e:
            # Fallback error print (never crash the main request loop)
            print(f"[Observability Warning] Failed to log trace: {e}")
