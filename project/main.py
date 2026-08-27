"""FastAPI server and Web Chat UI for the Universal RAG Support Agent."""

import html
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import markdown
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.core import settings, StructuredTracer, RequestTrace, TenantConfig
from src.agent import AgentRunner, AgentResponse

app = FastAPI(
    title="Universal RAG Support Agent",
    description="Enterprise-grade RAG support agent with multi-tenant architecture.",
    version="1.0.0"
)

# Enable CORS for browser chat UI and API clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Multi-tenant runner pool & tracer
runners: Dict[str, AgentRunner] = {}
tracer = StructuredTracer()


def format_ai_response_to_html(raw_text: str) -> str:
    """
    Parse and beautify AI Markdown response into clean, styled HTML.
    Extracts citation blocks `[Sources: ...]` and displays them cleanly in a discrete footer.
    """
    if not raw_text:
        return ""
        
    extracted_badges = []
    
    def extract_citations(match):
        inner = match.group(1).strip()
        items = re.split(r'[,;]\s*', inner)
        for item in items:
            item = item.strip()
            item = re.sub(r'^(?:Sources:\s*)+', '', item).strip()
            if ('.md' in item or '.json' in item or '>' in item) and len(item) > 3:
                clean_item = html.escape(item)
                badge = f'<span class="inline-citation" data-citation="{clean_item}" title="Click to inspect source">📄 {clean_item}</span>'
                if badge not in extracted_badges:
                    extracted_badges.append(badge)
        return ''

    # Extract & strip inline citation tags from paragraph body
    clean_text = re.sub(r'`?\[(?:Sources:\s*)?([^\]]+)\]`?', extract_citations, raw_text)
    clean_text = re.sub(r'\s+([.,;:!?])', r'\1', clean_text)
    clean_text = re.sub(r'\s*,\s*$', '', clean_text.strip())
    
    # Parse Markdown to HTML
    html_output = markdown.markdown(
        clean_text,
        extensions=["extra", "nl2br", "sane_lists"]
    )
    
    # Append clean footer with citations if present
    if extracted_badges:
        badges_html = " ".join(extracted_badges)
        html_output += f'<div class="citations-footer"><span class="citations-label">Sources:</span> {badges_html}</div>'
        
    return html_output


def get_runner(tenant_id: str) -> AgentRunner:
    """Retrieve or lazily initialize an AgentRunner instance for a brand tenant."""
    if tenant_id not in runners:
        runners[tenant_id] = AgentRunner(tenant_id=tenant_id)
    return runners[tenant_id]


class ChatRequest(BaseModel):
    message: str = Field(..., description="Customer message")
    session_id: Optional[str] = Field(default=None, description="Conversational session ID")
    tenant_id: Optional[str] = Field(default=settings.DEFAULT_TENANT, description="Brand tenant ID")


class ChatResponse(BaseModel):
    answer: str
    html_answer: str
    sources: List[str]
    referenced_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    handoff_recommended: bool
    intent: str
    tool_called: Optional[str] = None
    session_id: str
    tenant_id: str
    trace_id: str
    duration_ms: float


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page chat interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h2>Universal RAG Agent Running</h2>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint returning service status."""
    return {"status": "healthy", "service": "universal-rag-agent", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/document")
async def get_document(
    filename: str = Query(..., description="Name of the Markdown or JSON document"),
    tenant_id: str = Query(default=settings.DEFAULT_TENANT, description="Brand tenant ID")
):
    """
    Retrieve full original document content and rendered HTML preview for inspection.
    """
    try:
        tenant_cfg = TenantConfig.load(tenant_id)
        
        # Check if requesting orders JSON database
        if "order" in filename.lower() and (filename.endswith(".json") or "json" in filename.lower()):
            file_path = tenant_cfg.orders_file
            if not file_path.exists():
                file_path = settings.BASE_DIR / "data" / "orders.json"
            if not file_path.exists():
                file_path = Path(__file__).resolve().parent.parent / "data" / "orders.json"
                
            raw_text = file_path.read_text(encoding="utf-8")
            return {
                "filename": "data/orders.json",
                "raw_text": raw_text,
                "html_text": f"<pre><code>{raw_text}</code></pre>"
            }

        clean_filename = filename.split(">")[0].strip()
        kb_dirs = [
            tenant_cfg.knowledge_base_dir,
            settings.BASE_DIR / "knowledge-base",
            Path(__file__).resolve().parent.parent / "knowledge-base"
        ]
        
        target_path = None
        for d in kb_dirs:
            if not d.exists():
                continue
            # Direct match
            p = d / clean_filename
            if p.exists():
                target_path = p
                break
            # Fuzzy match (e.g. international, warranty, returns)
            keywords = [w for w in clean_filename.lower().replace(".md", "").split("-") if len(w) > 3]
            for candidate in d.glob("*.md"):
                if candidate.name.lower() == clean_filename.lower():
                    target_path = candidate
                    break
                if keywords and any(kw in candidate.name.lower() for kw in keywords):
                    target_path = candidate
                    break
            if target_path:
                break
                
        if not target_path or not target_path.exists():
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
            
        raw_text = target_path.read_text(encoding="utf-8")
        html_text = markdown.markdown(raw_text, extensions=["extra", "nl2br", "sane_lists"])
        
        return {
            "filename": target_path.name,
            "raw_text": raw_text,
            "html_text": html_text
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Primary chat endpoint for conversational customer support.
    
    Steps:
    1. Validate non-empty user input message.
    2. Extract or generate session_id and trace_id.
    3. Route request to tenant's AgentRunner.
    4. Parse and beautify raw Markdown response into semantic HTML.
    5. Record structured JSON interaction trace in logs/.
    6. Return JSON payload containing raw answer, beautified HTML, and referenced chunks.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    tenant_id = request.tenant_id or settings.DEFAULT_TENANT
    trace_id = str(uuid.uuid4())
    
    # Step 3: Run agent graph
    runner = get_runner(tenant_id)
    agent_res: AgentResponse = runner.chat(user_message=request.message, session_id=session_id)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    
    # Step 4: Parse & beautify markdown response
    html_answer = format_ai_response_to_html(agent_res.answer)
    
    # Step 5: Log structured observability trace
    tracer.log_interaction(RequestTrace(
        trace_id=trace_id,
        session_id=session_id,
        tenant_id=tenant_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        user_message=request.message,
        conversation_history=agent_res.conversation_history,
        intent=agent_res.intent,
        retrieved_sources=agent_res.sources,
        tool_called=agent_res.tool_called,
        final_response=agent_res.answer,
        handoff_recommended=agent_res.handoff_recommended,
        duration_ms=duration_ms
    ))
    
    # Step 6: Return response
    return ChatResponse(
        answer=agent_res.answer,
        html_answer=html_answer,
        sources=agent_res.sources,
        referenced_chunks=agent_res.referenced_chunks,
        handoff_recommended=agent_res.handoff_recommended,
        intent=agent_res.intent,
        tool_called=agent_res.tool_called,
        session_id=session_id,
        tenant_id=tenant_id,
        trace_id=trace_id,
        duration_ms=duration_ms
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
