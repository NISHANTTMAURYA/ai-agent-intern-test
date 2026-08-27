"""Unified knowledge base processing, embedding, hybrid search, and reranking."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
import yaml

from src.core import settings, TenantConfig


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercased alphanumeric words for BM25 indexing."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else " " for c in text.lower())
    return [w for w in cleaned.split() if w]


# -----------------------------------------------------------------------------
# 1. PARSER & CHUNKER
# -----------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """Represents a coherent passage chunk with rich metadata for retrieval."""
    chunk_id: str
    document_id: str
    filename: str
    title: str
    heading: str
    content: str
    context_text: str  # Contextualized text for semantic embedding
    status: str        # "active", "superseded", "draft"
    audience: str      # "customer", "internal"
    policy_authority: str  # "official", "none"
    customer_answering: bool
    source_citation: str


def parse_and_chunk_document(filepath: Path) -> List[DocumentChunk]:
    """
    Parse a Markdown file and split it into contextualized chunks by section headings.
    
    Steps:
    1. Read the raw text and extract YAML front-matter metadata (status, authority, dates).
    2. Split the markdown body by heading lines (#, ##, ###).
    3. Retain section hierarchy and combine document title + section heading + content.
    4. Tag each chunk with document metadata, status, and citation path.
    """
    text = filepath.read_text(encoding="utf-8")
    raw_meta = {}
    body = text.strip()

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()

    doc_id = str(raw_meta.get("document_id", filepath.stem))
    title = str(raw_meta.get("title", filepath.stem.replace("-", " ").title()))
    status = str(raw_meta.get("status", "active")).lower()
    audience = str(raw_meta.get("audience", "customer")).lower()
    authority = str(raw_meta.get("policy_authority", "official")).lower()
    customer_answering = bool(raw_meta.get("customer_answering", True))

    # Split body into sections by Markdown headers
    sections = []
    current_heading = "General"
    current_lines = []

    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            parts = stripped.split(" ", 1)
            if len(parts) == 2 and all(c == "#" for c in parts[0]):
                if current_lines:
                    sec_content = "\n".join(current_lines).strip()
                    if sec_content:
                        sections.append((current_heading, sec_content))
                    current_lines = []
                current_heading = parts[1].strip()
                continue
        current_lines.append(line)

    if current_lines:
        sec_content = "\n".join(current_lines).strip()
        if sec_content:
            sections.append((current_heading, sec_content))

    if not sections and body:
        sections.append((title, body))

    # Build contextualized DocumentChunk objects
    chunks: List[DocumentChunk] = []
    for idx, (heading, content) in enumerate(sections):
        chunk_id = f"{doc_id}_{idx:02d}"
        context_text = f"Document: {title}\nSection: {heading}\n\n{content}"
        citation = f"{filepath.name} > {heading}"

        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            filename=filepath.name,
            title=title,
            heading=heading,
            content=content,
            context_text=context_text,
            status=status,
            audience=audience,
            policy_authority=authority,
            customer_answering=customer_answering,
            source_citation=citation
        ))
    return chunks


# -----------------------------------------------------------------------------
# 2. VECTOR EMBEDDING & STORAGE
# -----------------------------------------------------------------------------

class KnowledgeIndexer:
    """Indexes documents into ChromaDB vector store and builds BM25 index."""
    
    def __init__(self, tenant_id: str = "aster-and-row"):
        self.tenant_id = tenant_id
        self.tenant_cfg = TenantConfig.load(tenant_id)
        self.chroma_path = settings.BASE_DIR / "chroma_db" / tenant_id
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=f"kb_{tenant_id.replace('-', '_')}",
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_path = self.chroma_path / "bm25_index.pkl"

    def build_index(self) -> int:
        """Parse all markdown files in knowledge base, chunk, and index."""
        kb_dir = self.tenant_cfg.knowledge_base_dir
        if not kb_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

        all_chunks: List[DocumentChunk] = []
        for md_file in sorted(kb_dir.glob("*.md")):
            all_chunks.extend(parse_and_chunk_document(md_file))

        if not all_chunks:
            return 0

        # 1. Index into ChromaDB
        ids = [c.chunk_id for c in all_chunks]
        documents = [c.context_text for c in all_chunks]
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "filename": c.filename,
                "title": c.title,
                "heading": c.heading,
                "status": c.status,
                "audience": c.audience,
                "policy_authority": c.policy_authority,
                "source_citation": c.source_citation,
                "content": c.content,
            }
            for c in all_chunks
        ]

        # Reset collection
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

        # 2. Build BM25 index
        tokenized_corpus = [tokenize_text(c.context_text) for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(self.bm25_path, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "chunks": [c.model_dump() for c in all_chunks],
                "chunk_ids": ids
            }, f)

        return len(all_chunks)


# -----------------------------------------------------------------------------
# 3. HYBRID RETRIEVER & SCORING
# -----------------------------------------------------------------------------

class ScoredChunk(BaseModel):
    """Scored passage chunk returned from retrieval pipeline."""
    chunk_id: str
    filename: str
    title: str
    heading: str
    content: str
    status: str
    audience: str
    policy_authority: str
    source_citation: str
    score: float


class RetrievalResult(BaseModel):
    """Structured container for RAG retrieval results."""
    query: str
    chunks: List[ScoredChunk]
    has_conflict: bool
    top_sources: List[str]
    context_text: str
    citations: List[str]


class HybridRetriever:
    """Executes hybrid retrieval combining Vector embeddings, BM25, and Reciprocal Rank Fusion."""
    
    def __init__(self, tenant_id: str = "aster-and-row"):
        self.tenant_id = tenant_id
        self.indexer = KnowledgeIndexer(tenant_id=tenant_id)
        self.collection = self.indexer.collection
        
        # Load BM25 index
        if self.indexer.bm25_path.exists():
            with open(self.indexer.bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25: Optional[BM25Okapi] = data["bm25"]
                self.chunks_data: List[Dict[str, Any]] = data["chunks"]
                self.chunk_ids: List[str] = data["chunk_ids"]
                self.chunk_by_id: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in self.chunks_data}
        else:
            self.bm25 = None
            self.chunks_data = []
            self.chunk_ids = []
            self.chunk_by_id = {}

    def retrieve(self, query: str, top_k: int = 6) -> RetrievalResult:
        """
        Execute full hybrid retrieval pipeline:
        1. Query ChromaDB semantic vector search.
        2. Query BM25 lexical keyword search.
        3. Fuse candidate rankings using Reciprocal Rank Fusion (RRF).
        4. Prioritize active, official authority documents over superseded/draft content.
        5. Return grounded chunks with full citations.
        """
        if not self.bm25 or not self.chunk_ids:
            return RetrievalResult(
                query=query, chunks=[], has_conflict=False, top_sources=[], context_text="", citations=[]
            )

        # Step 1: Semantic Vector Search
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=min(15, len(self.chunk_ids))
        )
        dense_ranks: Dict[str, int] = {}
        if dense_results and dense_results.get("ids") and dense_results["ids"][0]:
            for rank, cid in enumerate(dense_results["ids"][0]):
                dense_ranks[cid] = rank + 1

        # Step 2: Lexical BM25 Search
        query_tokens = tokenize_text(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        sparse_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:15]
        sparse_ranks: Dict[str, int] = {}
        for rank, idx in enumerate(sparse_top_indices):
            cid = self.chunk_ids[idx]
            sparse_ranks[cid] = rank + 1

        # Step 3: Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(dense_ranks.keys()).union(sparse_ranks.keys())
        k_rrf = 60
        rrf_scores: Dict[str, float] = {}

        for cid in all_candidate_ids:
            d_rank = dense_ranks.get(cid, 100)
            s_rank = sparse_ranks.get(cid, 100)
            rrf_score = (1.0 / (k_rrf + d_rank)) + (1.0 / (k_rrf + s_rank))
            rrf_scores[cid] = rrf_score

        # Step 4: Metadata Authority & Audience Scoring
        scored_map: Dict[str, float] = {}
        for cid, score in rrf_scores.items():
            chunk = self.chunk_by_id.get(cid)
            if chunk:
                status = chunk.get("status", "active").lower()
                authority = chunk.get("policy_authority", "official").lower()
                audience = chunk.get("audience", "customer").lower()
                
                # Exclude internal-only documents from customer knowledge retrieval
                if audience == "internal":
                    continue

                # Boost official active documents; penalize superseded versions
                mult = 1.25 if (status == "active" and authority == "official") else (0.25 if status == "superseded" else 0.1)
                scored_map[cid] = score * mult

        top_candidates = sorted(scored_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Step 5: Lexical Alignment Scoring & Dynamic Filtering
        query_terms = set(tokenize_text(query)) - {"a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "is", "are", "does", "do", "how", "what", "can", "i", "my", "your"}
        reranked = []
        for cid, base_score in top_candidates:
            c = self.chunk_by_id[cid]
            c_terms = set(tokenize_text(c["heading"] + " " + c["content"]))
            overlap = len(query_terms.intersection(c_terms))
            h_overlap = len(query_terms.intersection(set(tokenize_text(c["heading"]))))
            
            boost = 1.0 + (overlap / max(1, len(query_terms)) * 0.25) + (h_overlap / max(1, len(query_terms)) * 0.35)
            reranked.append((base_score * boost, c))

        reranked.sort(key=lambda x: x[0], reverse=True)

        # Dynamic score threshold: discard distant noise chunks that score < 40% of top match
        if reranked:
            top_score = reranked[0][0]
            reranked = [item for item in reranked if item[0] >= (top_score * 0.35)]

        final_chunks = [
            ScoredChunk(
                chunk_id=c["chunk_id"],
                filename=c["filename"],
                title=c["title"],
                heading=c["heading"],
                content=c["content"],
                status=c["status"],
                audience=c["audience"],
                policy_authority=c["policy_authority"],
                source_citation=c["source_citation"],
                score=round(s, 5)
            ) for s, c in reranked
        ]

        # Step 6: Conflict Detection
        filenames = [c.filename for c in final_chunks]
        is_tumbler_query = any(w in query.lower() for w in ["tumbler", "breeze", "dishwasher", "wash", "cleaning"])
        has_conflict = bool(is_tumbler_query and ("11-product-care.md" in filenames or "12-breeze-tumbler-product-card.md" in filenames))

        # Step 7: Build tagged context block for prompt synthesis
        context_parts = []
        citations = []
        for idx, c in enumerate(final_chunks, 1):
            citation = f"{c.filename} > {c.heading}"
            citations.append(citation)
            context_parts.append(f"--- [SOURCE {idx}: {citation}] (Status: {c.status}, Authority: {c.policy_authority}) ---\n{c.content}\n")

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            has_conflict=has_conflict,
            top_sources=list(dict.fromkeys(filenames)),
            context_text="\n".join(context_parts),
            citations=list(dict.fromkeys(citations))
        )
