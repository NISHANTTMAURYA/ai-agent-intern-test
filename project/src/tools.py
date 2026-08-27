"""Agent tools: privacy-safe order lookup and knowledge-base RAG wrapper."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core import settings, TenantConfig
from src.knowledge import HybridRetriever, RetrievalResult


def format_human_date(date_str: Optional[str]) -> str:
    """Format ISO date strings (e.g. '2026-08-22') into human-friendly format 'August 22, 2026'."""
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str.split("T")[0], "%Y-%m-%d")
        return dt.strftime("%B %d, %Y").replace(" 0", " ")
    except Exception:
        return date_str


class OrderItemSafe(BaseModel):
    """Sanitized item record safe for customer disclosure."""
    name: str
    quantity: int
    final_sale: bool = False


class OrderRecordSafe(BaseModel):
    """Customer-safe order status record with zero confidential internal fields."""
    order_id: str
    status: str
    membership_tier: str
    items: List[OrderItemSafe]
    placed_at: str
    status_updated_at: Optional[str] = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    requires_human_handoff: bool = False


class OrderLookupResponse(BaseModel):
    """Standard response model for order lookup tool execution."""
    success: bool
    status: str  # "found", "not_found", "missing_id"
    message: str
    order: Optional[OrderRecordSafe] = None
    requires_human_handoff: bool = False


class OrderLookupTool:
    """Order status lookup with input normalization, privacy sanitization, and status precedence."""
    
    def __init__(self, tenant_id: str = "aster-and-row"):
        self.tenant_id = tenant_id
        config = TenantConfig.load(tenant_id)
        self.orders_path = config.orders_file
        self.orders_data: Dict[str, dict] = {}
        self._load_orders()
        
    def _load_orders(self) -> None:
        """Load and index orders from tenant orders.json dataset."""
        if not self.orders_path.exists():
            return
        with open(self.orders_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for order in data.get("orders", []):
                oid = order.get("order_id", "").strip().upper()
                if oid:
                    self.orders_data[oid] = order

    @staticmethod
    def normalize_order_id(input_text: str) -> Optional[str]:
        """
        Extract and normalize an order ID dynamically.
        Handles: 'ORD-1007', 'ord_1007', 'ord 1007', 'order 10101', '#1007', 'ORD1007', or standalone '1007'.
        Avoids false positives on regular numbers like years (2026), postal codes (90210), or item counts (1000).
        """
        if not input_text:
            return None
            
        import re
        
        # 1. Match explicit ORD prefix variants: ORD-1007, ord_1007, ord 1007, ord1007
        m = re.search(r'\bORD[-_ ]?(\d{4,6})\b', input_text, re.IGNORECASE)
        if m:
            return f"ORD-{m.group(1)}"
            
        # 2. Match order/package keyword or hashtag before digits: "order 1007", "order #1007", "package 1007"
        m = re.search(r'\b(?:order|package)\s*(?:#|no\.?|id|number)?\s*(\d{4,6})\b', input_text, re.IGNORECASE)
        if m:
            return f"ORD-{m.group(1)}"
            
        m = re.search(r'#(\d{4,6})\b', input_text)
        if m:
            return f"ORD-{m.group(1)}"

        # 3. If the entire user input is strictly just 4-6 digits (e.g. user typed only "1007")
        clean_input = input_text.strip()
        if clean_input.isdigit() and len(clean_input) in [4, 5, 6]:
            return f"ORD-{clean_input}"

        return None

    def lookup(self, order_id_or_query: str) -> OrderLookupResponse:
        """
        Perform a safe lookup of an order by ID.
        
        Steps:
        1. Validate input and extract normalized Order ID.
        2. Query in-memory indexed dataset.
        3. If not found, return safe failure message and flag for human handoff.
        4. Apply Privacy Sanitization: NEVER expose email, address, internal notes, or risk scores.
        5. Apply Status Precedence: Strip stale carrier & ETA estimates on cancelled or returned orders.
        6. Return typed OrderLookupResponse with safe attributes only.
        """
        if not order_id_or_query or not order_id_or_query.strip():
            return OrderLookupResponse(
                success=False,
                status="missing_id",
                message="Please provide your order ID (e.g. ORD-1007) so I can check its status for you.",
                requires_human_handoff=False
            )
            
        normalized_id = self.normalize_order_id(order_id_or_query)
        if not normalized_id:
            clean_input = order_id_or_query.strip().upper()
            if "ORD" in clean_input or clean_input.isdigit():
                digits = "".join(c for c in clean_input if c.isdigit())
                normalized_id = f"ORD-{digits}" if digits else clean_input
            else:
                return OrderLookupResponse(
                    success=False,
                    status="missing_id",
                    message="Please provide your order ID (for example, ORD-1007) to look up your order status.",
                    requires_human_handoff=False
                )
                
        raw_order = self.orders_data.get(normalized_id)
        if not raw_order:
            return OrderLookupResponse(
                success=False,
                status="not_found",
                message=f"Order {normalized_id} was not found in our system. Please check the order number or contact support for help.",
                requires_human_handoff=True
            )
            
        status = raw_order.get("status", "unknown")
        
        # Privacy sanitization: items list only
        items = [
            OrderItemSafe(
                name=item.get("name", "Unknown Item"),
                quantity=item.get("quantity", 1),
                final_sale=item.get("final_sale", False)
            )
            for item in raw_order.get("items", [])
        ]
        
        # Status precedence: cancelled and returned orders NEVER report carrier/tracking/ETA
        if status in ["cancelled", "returned"]:
            carrier = None
            tracking_number = None
            estimated_delivery = None
        else:
            carrier = raw_order.get("carrier")
            tracking_number = raw_order.get("tracking_number")
            estimated_delivery = raw_order.get("estimated_delivery")
            
        safe_record = OrderRecordSafe(
            order_id=normalized_id,
            status=status,
            membership_tier=raw_order.get("membership_tier", "standard"),
            items=items,
            placed_at=raw_order.get("placed_at", ""),
            status_updated_at=raw_order.get("status_updated_at"),
            shipped_at=raw_order.get("shipped_at"),
            delivered_at=raw_order.get("delivered_at"),
            carrier=carrier,
            tracking_number=tracking_number,
            estimated_delivery=estimated_delivery,
            customer_safe_message=raw_order.get("customer_safe_message"),
            requires_human_handoff=(status in ["exception", "delayed", "damaged"])
        )
        
        # Format date to human friendly format if standard YYYY-MM-DD
        formatted_eta = estimated_delivery
        if estimated_delivery:
            try:
                from datetime import datetime
                dt = datetime.strptime(estimated_delivery.strip(), "%Y-%m-%d")
                formatted_eta = dt.strftime("%B %-d, %Y") if hasattr(dt, 'strftime') else estimated_delivery
            except Exception:
                try:
                    from datetime import datetime
                    dt = datetime.strptime(estimated_delivery.strip(), "%Y-%m-%d")
                    formatted_eta = dt.strftime("%B %d, %Y").replace(" 0", " ")
                except Exception:
                    formatted_eta = estimated_delivery

        # Construct dynamic, comprehensive customer-safe message
        msg_parts = [f"Order {normalized_id} has an official status of {status}."]
        if status == "shipped":
            if carrier and tracking_number:
                msg_parts.append(f"It is currently in transit with {carrier} under tracking number {tracking_number}.")
            elif carrier:
                msg_parts.append(f"It is currently in transit with {carrier}.")
            if formatted_eta:
                msg_parts.append(f"The estimated delivery date is {formatted_eta}.")
            else:
                msg_parts.append("A specific delivery estimate is unavailable at this time.")
        elif status == "cancelled":
            msg_parts.append("The order was cancelled and it will not be shipped or delivered.")
        elif status == "returned":
            msg_parts.append("The order has been returned.")
        elif status == "delivered":
            if safe_record.delivered_at:
                msg_parts.append(f"Delivered on {safe_record.delivered_at}.")
            else:
                msg_parts.append("The package has been delivered.")
        elif safe_record.customer_safe_message:
            msg_parts.append(safe_record.customer_safe_message)

        if safe_record.requires_human_handoff:
            msg_parts.append("I recommend human assistance for this order inquiry.")

        full_message = " ".join(msg_parts)

        return OrderLookupResponse(
            success=True,
            status="found",
            message=full_message,
            order=safe_record,
            requires_human_handoff=safe_record.requires_human_handoff
        )


class RAGTool:
    """Knowledge-base retrieval tool combining dense and sparse search."""
    
    def __init__(self, tenant_id: str = "aster-and-row"):
        self.tenant_id = tenant_id
        self.retriever = HybridRetriever(tenant_id=tenant_id)
        
    def query(self, search_query: str) -> RetrievalResult:
        """Execute knowledge retrieval for the query string."""
        return self.retriever.retrieve(query=search_query, top_k=6)
