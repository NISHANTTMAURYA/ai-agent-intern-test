"""CLI tool to ingest, parse, chunk, and index knowledge base documents."""

import argparse
import sys
from pathlib import Path
from src.core import settings, TenantConfig
from src.knowledge import KnowledgeIndexer


def run_ingestion(tenant_id: str = "aster-and-row") -> int:
    """
    Execute end-to-end ingestion pipeline for a brand tenant.
    
    Steps:
    1. Load tenant configuration and resolve knowledge base folder.
    2. Instantiate KnowledgeIndexer for the tenant.
    3. Parse, chunk, embed into ChromaDB, and build the BM25 index.
    4. Print summary metrics and return total chunks indexed.
    """
    config = TenantConfig.load(tenant_id)
    kb_dir = config.knowledge_base_dir
    
    print(f"\n=======================================================")
    print(f"  Ingesting Knowledge Base for Tenant: [{tenant_id}]")
    print(f"  Source Directory: {kb_dir}")
    print(f"=======================================================\n")
    
    if not kb_dir.exists():
        print(f"❌ Error: Knowledge base directory not found: {kb_dir}")
        sys.exit(1)
        
    indexer = KnowledgeIndexer(tenant_id=tenant_id)
    count = indexer.index_tenant(kb_dir=kb_dir)
    print(f"✅ Successfully indexed {count} chunks for tenant [{tenant_id}]!\n")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge base documents.")
    parser.add_argument("--tenant", type=str, default=settings.DEFAULT_TENANT, help="Tenant ID (default: aster-and-row)")
    args = parser.parse_args()
    run_ingestion(args.tenant)
