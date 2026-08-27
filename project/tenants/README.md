# Multi-Tenant Brand Guide

This directory manages multi-brand tenant configurations for the Universal RAG Support Agent.

---

## 📁 Directory Structure

```text
tenants/
├── README.md                  # Multi-tenant guide (this file)
├── aster-and-row/             # Default brand tenant
│   ├── tenant.yaml            # Brand metadata, persona & paths
│   ├── kb/                    # (Optional) Tenant-specific knowledge base docs
│   └── orders.json            # (Optional) Tenant-specific mock orders
└── [new-brand-id]/            # Add new brands here
    ├── tenant.yaml
    ├── kb/
    └── orders.json
```

---

## 🚀 How to Add a New Brand Tenant (3 Steps)

### Step 1: Create Tenant Directory & YAML Config
Create `tenants/{brand_id}/tenant.yaml`:

```yaml
tenant_id: "acme-outdoors"
brand_name: "Acme Outdoors"
support_email: "support@acmeoutdoors.com"
description: "Outdoor apparel and gear company"
persona: "Friendly, adventurous, concise customer support guide"

paths:
  knowledge_base_dir: "tenants/acme-outdoors/kb"
  orders_file: "tenants/acme-outdoors/orders.json"

guardrails:
  enforce_privacy: true
  human_handoff_threshold: 0.15
  require_source_citations: true
```

### Step 2: Add Brand Knowledge Base & Orders
1. Drop markdown policy files into `tenants/{brand_id}/kb/` (e.g. `01-returns.md`, `02-warranty.md`).
2. Drop order records into `tenants/{brand_id}/orders.json`.

### Step 3: Index the New Brand
Run the ingestion script for the new brand:

```bash
python ingest.py --tenant acme-outdoors
```

This creates an isolated ChromaDB collection and BM25 index under `chroma_db/acme-outdoors/`.

---

## 🌐 Querying a Tenant via API

Pass the `tenant_id` in the API payload:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is your return policy?",
    "tenant_id": "acme-outdoors"
  }'
```
