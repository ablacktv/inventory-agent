"""Supply Chain Analytics Inventory Decision Agent.

Architecture Pipeline:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Input Extraction:
- Uses the Gemini API (via GEMINI_API_KEY from environment or Streamlit secrets, never hardcoded).
- Fully falls back to an enhanced deterministic NLP & regex extractor if no API key is available or on error.
- Recognizes all common inventory terminology, ordering cost synonyms, and model-specific nuances.
- Strictly keeps all numerical calculations in deterministic Python tools in inventory_tools.py.
- Never invents missing numerical inputs.
- Thoroughly examines the entire question before declaring information missing.
"""

from dataclasses import dataclass, field
import json
import os
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import inventory_tools


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment variable or Streamlit secrets (never hardcoded)."""
    # 1. Check environment variable
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            s_key = st.secrets["GEMINI_API_KEY"]
            if s_key and str(s_key).strip():
                return str(s_key).strip()
    except Exception:
        pass

    return None


@dataclass
class AgentStep:
    """Represents a single step in the decision agent pipeline."""
    step_name: str
    status: str
    details: Dict[str, Any]
    description: str


@dataclass
class InventoryDecisionResult:
    """Complete output artifact returned by the Inventory Decision Agent."""
    customer_question: str
    identified_model: str
    model_display_name: str
    model_confidence: float
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    selected_tool: str
    calculation_results: Optional[Dict[str, Any]]
    verification: Optional[Dict[str, Any]]
    explanation: str
    execution_trace: List[AgentStep] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    extraction_method: str = "Deterministic Regex NLP"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization or API output."""
        return {
            "customer_question": self.customer_question,
            "identified_model": self.identified_model,
            "model_display_name": self.model_display_name,
            "model_confidence": self.model_confidence,
            "extraction_method": self.extraction_method,
            "extracted_inputs": self.extracted_inputs,
            "missing_inputs": self.missing_inputs,
            "selected_tool": self.selected_tool,
            "calculation_results": self.calculation_results,
            "verification": self.verification,
            "explanation": self.explanation,
            "execution_trace": [
                {
                    "step_name": s.step_name,
                    "status": s.status,
                    "details": s.details,
                    "description": s.description,
                }
                for s in self.execution_trace
            ],
            "success": self.success,
            "error_message": self.error_message,
        }


class SupplyChainInventoryAgent:
    """Agentic decision engine implementing the 7-step inventory analytics workflow."""

    MODEL_EOQ = "EOQ"
    MODEL_QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    MODEL_SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    MODEL_FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    MODEL_NEWSVENDOR = "NEWSVENDOR"

    MODEL_DISPLAY_NAMES = {
        MODEL_EOQ: "Economic Order Quantity (EOQ)",
        MODEL_QUANTITY_DISCOUNT: "Quantity Discount Analysis",
        MODEL_SAFETY_STOCK_ROP: "Safety Stock & Reorder Point (ROP)",
        MODEL_FIXED_PERIOD_REVIEW: "Fixed-Period Review (Periodic Review / Base Stock)",
        MODEL_NEWSVENDOR: "Newsvendor Model (Single-Period / Perishable)",
    }

    TOOL_MAPPING = {
        MODEL_EOQ: "calculate_eoq",
        MODEL_QUANTITY_DISCOUNT: "calculate_quantity_discount",
        MODEL_SAFETY_STOCK_ROP: "calculate_rop",
        MODEL_FIXED_PERIOD_REVIEW: "calculate_periodic_review",
        MODEL_NEWSVENDOR: "calculate_newsvendor",
    }

    # Ordering cost synonyms recognized across all models
    ORDERING_COST_SYNONYMS = [
        "ordering cost",
        "order cost",
        "setup cost",
        "setup expense",
        "cost per order",
        "cost of placing an order",
        "ordering/setup cost",
        "ordering and setup cost",
        "cost to place an order",
        "order fee",
        "replenishment cost per order",
        "ordering expense",
    ]

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------------
    # STEP 1 & 2 VIA GEMINI API (WITH FALLBACK)
    # ------------------------------------------------------------------------
    def extract_with_gemini(
        self, question: str, api_key: str
    ) -> Optional[Tuple[str, float, Dict[str, Any], List[str]]]:
        """Use the Gemini API to identify the model and extract structured numeric inputs.

        Returns:
            Tuple of (identified_model, confidence, extracted_inputs, missing_inputs) or None if error.
        """
        prompt = f"""You are an expert supply chain analytics inventory decision agent.
Analyze the following natural-language customer question and extract the inventory model and its numerical parameters.

Customer Question:
"{question}"

Supported Inventory Models:
1. "EOQ" - Economic Order Quantity (Continuous review, deterministic demand).
   - Parameters: demand (annual demand D), order_cost (ordering/setup cost S), holding_cost (unit holding cost H per year), holding_rate (carrying cost % h), unit_cost (purchase cost C), lead_time_days.
   - Ordering cost synonyms: "ordering cost", "order cost", "setup cost", "setup expense", "cost per order", "cost of placing an order", "ordering/setup cost".
2. "QUANTITY_DISCOUNT" - Quantity Discount Analysis (All-units price breaks).
   - Parameters: demand (annual demand), order_cost (ordering cost), holding_rate (% of unit price) OR holding_cost (fixed holding cost), tiers (list of objects with "min_qty", "max_qty" (null if open-ended), "price").
3. "SAFETY_STOCK_ROP" - Safety Stock and Reorder Point (Continuous review stochastic).
   - Distinguish: avg_demand (d), std_demand (sigma_d), lead_time (L), std_lead_time (sigma_L), service_level (cycle service level 0 to 1), z_score.
4. "FIXED_PERIOD_REVIEW" - Fixed-Period Review (Periodic Review / Base Stock).
   - Distinguish: review_period (P), lead_time (L), avg_demand (d), std_demand (sigma_d), service_level (0 to 1), z_score, current_inventory (on-hand), on_order (in transit), backorders.
5. "NEWSVENDOR" - Single-Period Perishable Inventory.
   - Distinguish: selling_price (P), cost (purchase cost C), salvage_value (S), penalty_cost (shortage penalty), demand_mean (mu), demand_std (sigma).

Rules:
- DO NOT invent or fabricate numerical values. Only extract numbers explicitly stated in the customer question.
- Do not report a parameter as missing if it is already present in the question under any of its synonyms.
- Return ONLY valid JSON in this exact structure:
{{
  "model": "EOQ" | "QUANTITY_DISCOUNT" | "SAFETY_STOCK_ROP" | "FIXED_PERIOD_REVIEW" | "NEWSVENDOR",
  "confidence": 0.95,
  "extracted_inputs": {{
     ...
  }},
  "missing_inputs": ["param1", ...]
}}
"""
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw_text = response.text.strip()
            # Clean possible markdown fences
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text.strip())
            model = parsed.get("model", self.MODEL_EOQ)
            confidence = float(parsed.get("confidence", 0.95))
            inputs = parsed.get("extracted_inputs", {})
            missing = parsed.get("missing_inputs", [])
            return model, confidence, inputs, missing
        except Exception:
            # Fall back to deterministic extractor
            return None

    # ------------------------------------------------------------------------
    # STEP 1: IDENTIFY MODEL (DETERMINISTIC REGEX / KEYWORD ENGINE)
    # ------------------------------------------------------------------------
    def identify_model(self, question: str) -> Tuple[str, float, Dict[str, float]]:
        """Identify which inventory model applies to the natural language question.

        Uses weighted domain keywords and semantic signal indicators.
        """
        q = question.lower()

        scores: Dict[str, float] = {
            self.MODEL_EOQ: 0.0,
            self.MODEL_QUANTITY_DISCOUNT: 0.0,
            self.MODEL_SAFETY_STOCK_ROP: 0.0,
            self.MODEL_FIXED_PERIOD_REVIEW: 0.0,
            self.MODEL_NEWSVENDOR: 0.0,
        }

        # Quantity Discount cues
        if any(w in q for w in ["quantity discount", "price break", "volume discount", "discount schedule", "tier", "price tiers", "brackets", "price breaks"]):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 7.0
        if "discount" in q:
            scores[self.MODEL_QUANTITY_DISCOUNT] += 3.5
        if re.search(r"\d+\s*-\s*\d+\s*:", q) or re.search(r"\d+\s*\+\s*:", q):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 4.5

        # Fixed Period Review cues
        if any(w in q for w in ["fixed period", "fixed-period", "periodic review", "order-up-to", "order up to", "target level", "review period", "base stock", "p-system", "periodic system"]):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 7.0
        if "periodic" in q or "review interval" in q or re.search(r"(?:reviews?|reviewed|check(?:ed)?|orders?)\s+every", q):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 4.0
        if "on hand" in q and ("review" in q or "target" in q or "base stock" in q):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 3.0

        # Safety Stock & ROP cues
        if any(w in q for w in ["safety stock", "reorder point", "rop", "cycle service level", "lead time demand"]):
            scores[self.MODEL_SAFETY_STOCK_ROP] += 6.5
        if "service level" in q and ("reorder" in q or "safety" in q or "lead time" in q):
            scores[self.MODEL_SAFETY_STOCK_ROP] += 4.0
        if "standard deviation of demand" in q or "lead time variability" in q or "std dev" in q:
            scores[self.MODEL_SAFETY_STOCK_ROP] += 2.5

        # Newsvendor cues
        if any(w in q for w in ["newsvendor", "newsboy", "single period", "single-period", "perishable", "critical ratio", "salvage", "overage", "underage"]):
            scores[self.MODEL_NEWSVENDOR] += 7.5
        if "newspaper" in q or "spoilage" in q or "unsold units" in q or "end of day" in q or "seasonal" in q or "commemorative" in q or "event inventory" in q:
            scores[self.MODEL_NEWSVENDOR] += 4.5
        if "salvage value" in q or "scrap value" in q or "liquidated for" in q or "shortage cost" in q:
            scores[self.MODEL_NEWSVENDOR] += 3.5

        # EOQ cues
        if any(w in q for w in ["eoq", "economic order quantity", "optimal batch size", "optimal order quantity"]):
            scores[self.MODEL_EOQ] += 5.5
        # Ordering cost synonyms
        for syn in self.ORDERING_COST_SYNONYMS:
            if syn in q:
                scores[self.MODEL_EOQ] += 2.5
                break
        if "holding cost" in q or "carrying cost" in q or "cost to hold" in q:
            scores[self.MODEL_EOQ] += 2.0
        if "annual demand" in q or "yearly demand" in q:
            scores[self.MODEL_EOQ] += 2.0

        # Specificity tie-breakers
        if scores[self.MODEL_QUANTITY_DISCOUNT] > 0 and scores[self.MODEL_QUANTITY_DISCOUNT] >= scores[self.MODEL_EOQ]:
            scores[self.MODEL_QUANTITY_DISCOUNT] += 1.5

        if ("review period" in q or "reviews every" in q or "order up to" in q) and scores[self.MODEL_FIXED_PERIOD_REVIEW] > 0:
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 2.0

        if ("salvage" in q or "perishable" in q or "seasonal" in q) and scores[self.MODEL_NEWSVENDOR] > 0:
            scores[self.MODEL_NEWSVENDOR] += 2.0

        best_model = max(scores, key=lambda k: scores[k])
        max_score = scores[best_model]

        if max_score == 0.0:
            best_model = self.MODEL_EOQ
            confidence = 0.35
        else:
            total_score = sum(scores.values())
            confidence = min(0.99, round(max_score / (total_score if total_score > 0 else 1.0), 3))

        return best_model, confidence, scores

    # ------------------------------------------------------------------------
    # STEP 2: EXTRACT INPUTS (DETERMINISTIC REGEX / NLP SCANNER)
    # ------------------------------------------------------------------------
    def extract_inputs(self, question: str, model: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract structured numeric parameters from the natural language question.

        Examines the entire question with comprehensive synonym recognition and
        bidirectional prefix/suffix parsing before declaring any information missing.

        Returns:
            Tuple of (extracted_inputs_dict, missing_required_fields_list).
        """
        text = question
        inputs: Dict[str, Any] = {}
        num_pat = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

        def extract_first(patterns: List[str], target_type=float) -> Optional[Any]:
            for pat in patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    raw_str = (
                        match.group(1)
                        .replace("$", "")
                        .replace("%", "")
                        .strip()
                        .rstrip(".,")
                        .replace(",", "")
                    )
                    try:
                        return target_type(raw_str)
                    except (ValueError, TypeError):
                        continue
            return None

        # --------------------------------------------------------------------
        # A. DEMAND EXTRACTION (Annual, Daily, Weekly, Mean)
        # --------------------------------------------------------------------
        annual_demand = extract_first([
            r"(?:annual\s+demand|demand\s+per\s+year|year(?:ly)?\s+demand|annual\s+usage|annual\s+sales)[^\d]*(" + num_pat + ")",
            r"(?:demand|sales|requires)\s+(?:of|is|equals)?\s*(" + num_pat + r")\s*(?:units|items|pieces)?\s*(?:per\s+year|annually|\/year)",
            r"(" + num_pat + r")\s*units\s*(?:per\s+year|annually)",
            r"(?:demand|sales|annual\s+demand)\s*(?:is|of|equals)?\s*(" + num_pat + ")",
        ])

        daily_weekly_demand = extract_first([
            r"(?:daily\s+demand|demand\s+per\s+day|average\s+daily\s+demand|mean\s+daily\s+demand)[^\d]*(" + num_pat + ")",
            r"(" + num_pat + r")\s*(?:units)?\s*(?:per\s+day|\/day)",
            r"(?:weekly\s+demand|demand\s+per\s+week|average\s+weekly\s+demand)[^\d]*(" + num_pat + ")",
            r"(" + num_pat + r")\s*(?:units)?\s*(?:per\s+week|\/week)",
            r"(?:average\s+demand|mean\s+demand|demand\s+rate|expected\s+demand)\s*(?:is|of|equals)?\s*(" + num_pat + ")",
        ])

        demand_mean_explicit = extract_first([
            r"(?:normally\s+distributed\s+with\s+a\s+mean\s+of|with\s+a\s+mean\s+of|mean\s+of|mean\s+is)\s*(" + num_pat + ")",
            r"(?:mean|average|expected)\s*(?:demand|sales)?\s*(?:is|of|equals)?\s*(" + num_pat + ")",
        ])

        # --------------------------------------------------------------------
        # B. ORDERING / SETUP COST (Comprehensive Synonyms)
        # Recognizes:
        # - ordering cost
        # - order cost
        # - setup cost
        # - setup expense
        # - cost per order
        # - cost of placing an order
        # - ordering/setup cost
        # - ordering and setup cost
        # - cost to place an order
        # - order fee
        # - replenishment cost per order
        # --------------------------------------------------------------------
        order_cost = extract_first([
            # "ordering/setup cost is $60", "setup expense of $50", "cost of placing an order is $40"
            r"(?:ordering\s*\/\s*setup\s+cost|ordering\s+and\s+setup\s+cost|setup\s+expense|setup\s+cost|ordering\s+cost|order\s+cost|cost\s+of\s+placing\s+an\s+order|cost\s+to\s+place\s+an\s+order|cost\s+per\s+order|order\s+fee|replenishment\s+cost\s+per\s+order)[^\d$]*\$?\s*(" + num_pat + ")",
            # "$50 per order", "$60 to place an order", "$45 setup"
            r"\$\s*(" + num_pat + r")\s*(?:per\s+order|\/order|to\s+place\s+an\s+order|setup\s+expense|setup|ordering)",
            # "orders cost $50", "order fee is $25"
            r"(?:orders?|setup)\s+(?:costs?|expense|fee)\s*(?:is|of|equals)?\s*\$?\s*(" + num_pat + ")",
        ])

        # --------------------------------------------------------------------
        # C. HOLDING / CARRYING COST (Fixed Dollar vs Percentage)
        # --------------------------------------------------------------------
        holding_cost_unit = extract_first([
            r"(?:holding\s+cost|carrying\s+cost|cost\s+to\s+hold|cost\s+of\s+capital|carrying\s+expense|storage\s+cost)[^\d$]*\$?\s*(" + num_pat + r")\s*(?:per\s+unit|\/unit)(?!\s*%)",
            r"\$\s*(" + num_pat + r")\s*(?:per\s+unit\s+per\s+year|\/unit\/year|per\s+unit\s+to\s+hold|holding|carrying)(?!\s*%)",
            r"(?:holding\s+cost|carrying\s+cost|cost\s+to\s+hold)\s*(?:is|of|equals)?\s*\$?\s*(" + num_pat + r")(?!\s*%)",
        ])

        holding_pct = extract_first([
            r"(?:holding\s+cost|carrying\s+rate|holding\s+rate|carrying\s+cost|cost\s+of\s+capital|carrying\s+expense)[^\d%]*([\d\.]+)\s*%",
            r"([\d\.]+)\s*%\s*(?:holding|carrying|of\s+(?:item\s+)?cost|of\s+purchase\s+price|annual\s+holding)",
            r"(?:holding\s+rate|carrying\s+rate)\s*(?:is|of|equals)?\s*0\.(\d+)",
        ])
        if holding_pct is not None:
            if holding_pct > 1.0:
                holding_pct = holding_pct / 100.0

        # --------------------------------------------------------------------
        # D. UNIT PURCHASE PRICE / PRODUCT COST
        # --------------------------------------------------------------------
        unit_price = extract_first([
            r"(?:unit\s+price|purchase\s+price|price\s+per\s+unit|wholesale\s+price|cost\s+per\s+unit|unit\s+cost|item\s+cost|wholesale\s+cost|purchase\s+cost)[^\d$]*\$?\s*(" + num_pat + ")",
            r"(?:unit\s+costs?|item\s+costs?|price)[^\d$]*\$?\s*(" + num_pat + r")\s*(?:each|per\s+unit|\/unit|apiece)",
            r"\$\s*(" + num_pat + r")\s*(?:each|apiece)",
        ])

        # --------------------------------------------------------------------
        # E. LEAD TIME (Average & Standard Deviation)
        # --------------------------------------------------------------------
        lead_time = extract_first([
            r"(?:average\s+lead\s+time|mean\s+lead\s+time|replenishment\s+lead\s+time|supplier\s+lead\s+time|lead\s+time|delivery\s+time)[^\d]*(" + num_pat + r")\s*(?:days|weeks|months|periods)?",
            r"takes\s*(" + num_pat + r")\s*(?:days|weeks|periods)\s+to\s+arrive",
        ])

        std_lead_time = extract_first([
            r"(?:lead[-_\s]*time\s+standard\s+deviation|standard\s+deviation\s+of\s+lead\s+time|lead[-_\s]*time\s+std\s+dev|sigma_l)[^\d]*(" + num_pat + ")",
            r"(?:variability\s+in\s+lead\s+time)[^\d]*(" + num_pat + ")",
        ])

        # --------------------------------------------------------------------
        # F. DEMAND VARIABILITY (Standard Deviation)
        # --------------------------------------------------------------------
        std_demand = extract_first([
            r"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|demand\s+std\s+dev|std\s+dev\s+of\s+demand|sigma_d)[^\d]*(" + num_pat + ")",
            r"(?:with\s+a\s+standard\s+deviation\s+of|standard\s+deviation\s+of|standard\s+deviation\s+is)\s*(" + num_pat + ")",
            r"(?:standard\s+deviation|std\s+dev)[^\d]*(" + num_pat + ")",
        ])

        # --------------------------------------------------------------------
        # G. SERVICE LEVEL & Z-SCORE
        # --------------------------------------------------------------------
        service_level = extract_first([
            r"([\d\.]+)\s*%\s*(?:cycle\s+)?(?:service\s+level|csl|in[-_\s]*stock|fill\s+rate)",
            r"(?:service\s+level|cycle\s+service\s+level|csl|in[-_\s]*stock\s+probability|fill\s+rate)[^\d%]*([\d\.]+)\s*%",
            r"(?:service\s+level|cycle\s+service\s+level|csl)[^\d]*(0\.\d+)",
            r"(0\.\d+)\s*(?:cycle\s+)?(?:service\s+level|csl)",
        ])
        if service_level is not None:
            if service_level > 1.0:
                service_level = service_level / 100.0

        z_score = extract_first([
            r"(?:z[-_\s]*score|z[-_\s]*value|safety\s+factor\s*z?)[^\d]*([\d\.]+)",
        ])

        # --------------------------------------------------------------------
        # H. FIXED-PERIOD REVIEW SPECIFIC
        # (Review Period, Direct Safety Stock, On-Hand, On-Order, Backorders)
        # --------------------------------------------------------------------
        review_period = extract_first([
            r"(?:review\s+period|period\s+between\s+orders|review\s+interval|order\s+interval)[^\d]*(" + num_pat + r")\s*(?:days|weeks|months|periods)?",
            r"(?:reviews?|reviewed|check(?:ed|s)?|orders?|placed)\s+every\s*(" + num_pat + r")\s*(?:days|weeks|months|periods)?",
            r"every\s*(" + num_pat + r")\s*(?:days|weeks|months|periods)?\s*(?:review)?",
        ])

        direct_safety_stock = extract_first([
            r"(?:safety\s+stock\s+of|maintain\s+a\s+safety\s+stock\s+of)\s*(" + num_pat + r")\s*(?:units)?",
        ])

        current_inventory = extract_first([
            r"(" + num_pat + r")\s*(?:units\s+)?(?:on[-_\s]*hand|in\s+stock|available)",
            r"(?:current\s+inventory|on[-_\s]*hand\s+inventory|in\s+stock|currently\s+have)\s*(?:is|of|equals)?\s*(" + num_pat + ")",
            r"(?:currently\s+have|have)\s*(" + num_pat + r")\s*(?:units)?",
        ])

        on_order = extract_first([
            r"(" + num_pat + r")\s*(?:units\s+)?(?:on[-_\s]*order|in[-_\s]*transit|ordered)",
            r"(?:on[-_\s]*order|in[-_\s]*transit|pipeline\s+inventory|ordered)\s*(?:is|of|equals)?\s*(" + num_pat + ")",
        ])

        backorders = extract_first([
            r"(" + num_pat + r")\s*(?:units\s+)?(?:backorders?|backordered|unfilled)",
            r"(?:backorder(?:ed)?s?|unfilled\s+orders?)\s*(?:is|of|equals)?\s*(" + num_pat + ")",
        ])

        # --------------------------------------------------------------------
        # I. NEWSVENDOR SPECIFIC
        # (Selling Price, Purchase Cost, Salvage Value, Penalty/Shortage Cost)
        # --------------------------------------------------------------------
        selling_price = extract_first([
            r"(?:selling\s+price|retail\s+price|sells?\s+for|sale\s+price|revenue\s+per\s+unit)\s*(?:is|of|equals|for)?\s*\$?\s*(" + num_pat + ")",
            r"\$\s*(" + num_pat + r")\s*(?:retail|selling\s+price|sale\s+price|per\s+unit\s+selling|sells?\s+for)",
        ])

        purchase_cost = extract_first([
            r"(?:purchase\s+cost|unit\s+cost|wholesale\s+cost|costs?\s+us|acquisition\s+cost|production\s+cost|cost\s+per\s+unit)\s*(?:is|of|equals|for)?\s*\$?\s*(" + num_pat + ")",
            r"(?:costs?|wholesale)\s*\$?\s*(" + num_pat + r")\s*(?:each|to\s+buy|to\s+make|per\s+unit)",
        ])
        if purchase_cost is None:
            purchase_cost = unit_price

        salvage_value = extract_first([
            r"(?:salvage\s+value|scrap\s+value|clearance\s+price|liquidat\w*\s+for|liquidat\w*\s+value|unsold\s+for|salvage)\s*(?:is|of|equals|for)?\s*\$?\s*(" + num_pat + ")",
            r"\$\s*(" + num_pat + r")\s*(?:per\s+unit\s+)?(?:salvage|scrap|clearance|liquidation)\s*(?:value|price)?",
        ])

        penalty_cost = extract_first([
            r"(?:underage\s+cost|shortage\s+cost|penalty\s+cost|loss\s+of\s+goodwill|goodwill\s+penalty|stockout\s+penalty)\s*(?:is|of|equals|for)?\s*\$?\s*(" + num_pat + ")",
            r"\$\s*(" + num_pat + r")\s*(?:penalty|shortage\s+cost|goodwill\s+loss|underage)",
        ])

        # --------------------------------------------------------------------
        # J. QUANTITY DISCOUNT SCHEDULE EXTRACTION
        # --------------------------------------------------------------------
        discount_schedule = self._extract_discount_schedule(text)

        # --------------------------------------------------------------------
        # ASSEMBLE MODEL INPUTS & SYNTHESIZE DERIVED RELATIONSHIPS
        # --------------------------------------------------------------------
        if model == self.MODEL_EOQ:
            if annual_demand is not None:
                inputs["demand"] = annual_demand
                inputs["annual_demand"] = annual_demand
            elif daily_weekly_demand is not None:
                # If daily demand is given without annual, use it
                inputs["demand"] = daily_weekly_demand
                inputs["annual_demand"] = daily_weekly_demand

            if order_cost is not None:
                inputs["order_cost"] = order_cost

            if holding_cost_unit is not None:
                inputs["holding_cost"] = holding_cost_unit
                inputs["holding_cost_per_unit"] = holding_cost_unit
            elif holding_pct is not None and unit_price is not None:
                # Synthesize H = h * C so holding cost is not declared missing!
                inputs["holding_cost"] = holding_pct * unit_price
                inputs["holding_cost_per_unit"] = holding_pct * unit_price
                inputs["holding_rate"] = holding_pct
                inputs["holding_cost_pct"] = holding_pct
                inputs["unit_cost"] = unit_price
                inputs["unit_price"] = unit_price
            elif holding_pct is not None:
                inputs["holding_rate"] = holding_pct
                inputs["holding_cost_pct"] = holding_pct

            if unit_price is not None:
                inputs["unit_cost"] = unit_price
                inputs["unit_price"] = unit_price
            if lead_time is not None:
                inputs["lead_time_days"] = lead_time

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            if annual_demand is not None:
                inputs["demand"] = annual_demand
                inputs["annual_demand"] = annual_demand
            elif daily_weekly_demand is not None:
                inputs["demand"] = daily_weekly_demand
                inputs["annual_demand"] = daily_weekly_demand

            if order_cost is not None:
                inputs["order_cost"] = order_cost

            if holding_pct is not None:
                inputs["holding_rate"] = holding_pct
                inputs["holding_cost_pct"] = holding_pct
            elif holding_cost_unit is not None:
                inputs["holding_cost"] = holding_cost_unit
                inputs["fixed_holding_cost"] = holding_cost_unit

            if discount_schedule:
                inputs["tiers"] = discount_schedule
                inputs["discount_schedule"] = discount_schedule

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            demand_val = (
                daily_weekly_demand
                if daily_weekly_demand is not None
                else (demand_mean_explicit if demand_mean_explicit is not None else annual_demand)
            )
            if demand_val is not None:
                inputs["avg_demand"] = demand_val
            if lead_time is not None:
                inputs["lead_time"] = lead_time
            if std_demand is not None:
                inputs["std_demand"] = std_demand
            else:
                inputs["std_demand"] = 0.0
            if std_lead_time is not None:
                inputs["std_lead_time"] = std_lead_time
            else:
                inputs["std_lead_time"] = 0.0
            if service_level is not None:
                inputs["service_level"] = service_level
            if z_score is not None:
                inputs["z_score"] = z_score
            if unit_price is not None:
                inputs["unit_cost"] = unit_price
            if holding_pct is not None:
                inputs["annual_holding_rate"] = holding_pct

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            if review_period is not None:
                inputs["review_period"] = review_period
            if lead_time is not None:
                inputs["lead_time"] = lead_time
            demand_val = (
                daily_weekly_demand
                if daily_weekly_demand is not None
                else (demand_mean_explicit if demand_mean_explicit is not None else annual_demand)
            )
            if demand_val is not None:
                inputs["avg_demand"] = demand_val
            if std_demand is not None:
                inputs["std_demand"] = std_demand
            else:
                inputs["std_demand"] = 0.0
            if service_level is not None:
                inputs["service_level"] = service_level
            if z_score is not None:
                inputs["z_score"] = z_score
            if current_inventory is not None:
                inputs["current_inventory"] = current_inventory
            else:
                inputs["current_inventory"] = 0.0
            if on_order is not None:
                inputs["on_order"] = on_order
            else:
                inputs["on_order"] = 0.0
            if backorders is not None:
                inputs["backorders"] = backorders
            else:
                inputs["backorders"] = 0.0

        elif model == self.MODEL_NEWSVENDOR:
            if selling_price is not None:
                inputs["selling_price"] = selling_price
            if purchase_cost is not None:
                inputs["cost"] = purchase_cost
            if salvage_value is not None:
                inputs["salvage_value"] = salvage_value
            else:
                inputs["salvage_value"] = 0.0
            if penalty_cost is not None:
                inputs["penalty_cost"] = penalty_cost
            else:
                inputs["penalty_cost"] = 0.0

            demand_val = (
                demand_mean_explicit
                if demand_mean_explicit is not None
                else (daily_weekly_demand if daily_weekly_demand is not None else annual_demand)
            )
            if demand_val is not None:
                inputs["demand_mean"] = demand_val
            if std_demand is not None:
                inputs["demand_std"] = std_demand

        # --------------------------------------------------------------------
        # EXAMINE ENTIRE QUESTION BEFORE DECLARING INFORMATION MISSING
        # --------------------------------------------------------------------
        missing: List[str] = []

        if model == self.MODEL_EOQ:
            if "demand" not in inputs and "annual_demand" not in inputs:
                missing.append("annual_demand (D)")
            if "order_cost" not in inputs:
                missing.append("order_cost (S - ordering / setup cost)")
            if ("holding_cost" not in inputs and "holding_cost_per_unit" not in inputs) and not (
                ("holding_rate" in inputs or "holding_cost_pct" in inputs) and ("unit_cost" in inputs or "unit_price" in inputs)
            ):
                missing.append("holding_cost (H or carrying rate % with unit purchase price)")

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            if "demand" not in inputs and "annual_demand" not in inputs:
                missing.append("annual_demand (D)")
            if "order_cost" not in inputs:
                missing.append("order_cost (S - ordering / setup cost)")
            if ("holding_rate" not in inputs and "holding_cost_pct" not in inputs) and (
                "holding_cost" not in inputs and "fixed_holding_cost" not in inputs
            ):
                missing.append("holding_cost (holding rate % or fixed holding cost)")
            if ("tiers" not in inputs and "discount_schedule" not in inputs) or not inputs.get("tiers", inputs.get("discount_schedule")):
                missing.append("discount_schedule (tiers with quantity breaks and unit prices)")

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            if "avg_demand" not in inputs:
                missing.append("avg_demand (average demand rate d)")
            if "lead_time" not in inputs:
                missing.append("lead_time (average lead time L)")
            if (inputs.get("std_demand", 0.0) > 0 or inputs.get("std_lead_time", 0.0) > 0) and (
                "service_level" not in inputs and "z_score" not in inputs
            ):
                missing.append("service_level (target cycle service level CSL or z-score)")

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            if "review_period" not in inputs:
                missing.append("review_period (P - review interval)")
            if "lead_time" not in inputs:
                missing.append("lead_time (L - replenishment lead time)")
            if "avg_demand" not in inputs:
                missing.append("avg_demand (average demand rate d)")
            if inputs.get("std_demand", 0.0) > 0 and ("service_level" not in inputs and "z_score" not in inputs):
                missing.append("service_level (target service level or z-score)")

        elif model == self.MODEL_NEWSVENDOR:
            if "selling_price" not in inputs:
                missing.append("selling_price (P - selling / retail price)")
            if "cost" not in inputs:
                missing.append("cost (C - unit purchase / wholesale cost)")
            if "demand_mean" not in inputs:
                missing.append("demand_mean (expected / mean demand mu)")
            if "demand_std" not in inputs:
                missing.append("demand_std (demand standard deviation sigma)")

        return inputs, missing

    # ------------------------------------------------------------------------
    # STEP 2b: TIME-UNIT NORMALIZATION (DETERMINISTIC PYTHON)
    # ------------------------------------------------------------------------
    _PERIODS_PER_YEAR = {"day": 365.0, "week": 52.0, "month": 12.0, "quarter": 4.0, "year": 1.0}
    _DAYS_PER_PERIOD = {"day": 1.0, "week": 7.0, "month": 365.0 / 12.0, "quarter": 365.0 / 4.0, "year": 365.0}
    _PERIOD_WORDS = {
        "day": "day", "days": "day", "daily": "day",
        "week": "week", "weeks": "week", "weekly": "week",
        "month": "month", "months": "month", "monthly": "month",
        "quarter": "quarter", "quarters": "quarter", "quarterly": "quarter",
        "year": "year", "years": "year", "yearly": "year", "annual": "year", "annually": "year", "annum": "year",
    }

    @staticmethod
    def _num(s: str) -> float:
        return float(s.replace(",", "").rstrip(".,"))

    def _detect_demand_period(self, text: str) -> Optional[Tuple[float, str]]:
        """Return (stated_demand_value, period) when the question states a demand rate with a time unit."""
        num = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
        pw = r"(day|week|month|quarter|year|annum)"
        # "monthly demand is 1,000", "annual usage of 12,000"
        m = re.search(r"\b(daily|weekly|monthly|quarterly|annual|yearly)\s+(?:average\s+)?(?:demand|usage|sales|requirements?)\b[^\d$]{0,40}" + num, text, re.I)
        if m:
            return self._num(m.group(2)), self._PERIOD_WORDS[m.group(1).lower()]
        # "demand is 1,000 units per month", "sells 50 units a day", "25 units/day"
        m = re.search(r"(?<![\$\d.,])" + num + r"\s*(?:units?|items?|pieces?|cases?|boxes|pallets|bags|cartons)?\s*(?:per|/|a|an|each)\s*" + pw + r"\b", text, re.I)
        if m:
            ctx = text[max(0, m.start() - 60):m.start()].lower()
            if not re.search(r"(holding|carrying|storage|cost|price|\$)\s*[^.]*$", ctx):
                return self._num(m.group(1)), self._PERIOD_WORDS[m.group(2).lower()]
        # "demand per month is 1,000"
        m = re.search(r"(?:demand|usage|sales)\s+(?:per|each|a)\s+" + pw + r"\b[^\d$]{0,20}" + num, text, re.I)
        if m:
            return self._num(m.group(2)), self._PERIOD_WORDS[m.group(1).lower()]
        return None

    def _detect_holding_period(self, text: str) -> str:
        """Holding cost period; holding cost is annual unless another period is stated."""
        m = re.search(r"(?:holding|carrying|storage)[^.;]*?(?:per|/|a|each)\s*(?:unit\s*(?:per|/|a|each)\s*)?(day|week|month|quarter|year|annum)\b", text, re.I)
        if m:
            return self._PERIOD_WORDS[m.group(1).lower()]
        m = re.search(r"\b(daily|weekly|monthly|quarterly|annual|yearly)\s+(?:holding|carrying)", text, re.I)
        if m:
            return self._PERIOD_WORDS[m.group(1).lower()]
        return "year"

    def _detect_duration(self, text: str, keyword_regex: str) -> Optional[Tuple[float, str]]:
        num = r"(\d+(?:\.\d+)?)"
        m = re.search(keyword_regex + r"[^\d]{0,40}" + num + r"\s*(day|days|week|weeks|month|months|year|years)\b", text, re.I)
        if m:
            return float(m.group(1)), self._PERIOD_WORDS[m.group(2).lower()]
        return None

    def normalize_time_units(self, question: str, model: str, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Convert every rate to one consistent time basis before any tool is called.

        EOQ / Quantity Discount: demand is converted to the holding-cost period (annual by default).
        Safety Stock / Fixed-Period Review: lead time, its std dev and the review period are converted
        into the demand period (demand std dev is already per demand period).
        Values that were already converted (e.g. by Gemini) are left unchanged, so nothing is converted twice.
        """
        inputs = dict(inputs)
        notes: List[str] = []
        demand = self._detect_demand_period(question)

        if model in (self.MODEL_EOQ, self.MODEL_QUANTITY_DISCOUNT):
            if demand is None:
                return inputs, notes
            raw, d_per = demand
            h_per = self._detect_holding_period(question)
            if d_per == h_per:
                return inputs, notes
            factor = self._PERIODS_PER_YEAR[d_per] / self._PERIODS_PER_YEAR[h_per]
            converted = raw * factor
            for key in ("demand", "annual_demand"):
                v = inputs.get(key)
                if v is None:
                    continue
                if math.isclose(float(v), raw, rel_tol=1e-9):
                    inputs[key] = converted
                elif math.isclose(float(v), converted, rel_tol=1e-6):
                    pass  # already converted upstream
            if any(k in inputs for k in ("demand", "annual_demand")):
                notes.append(
                    f"Demand stated per {d_per} ({raw:,.4g}) but holding cost is per {h_per}: "
                    f"D = {raw:,.4g} x {factor:g} = {converted:,.4f} units per {h_per}."
                )
            return inputs, notes

        if model in (self.MODEL_SAFETY_STOCK_ROP, self.MODEL_FIXED_PERIOD_REVIEW):
            if demand is None:
                return inputs, notes
            _, d_per = demand
            checks = [
                ("lead_time", r"lead[\s-]*time"),
                ("review_period", r"(?:review\s+period|review\s+interval|reviews?\s+every|reviewed\s+every|every)"),
            ]
            lt = self._detect_duration(question, r"lead[\s-]*time")
            for key, kw in checks:
                found = self._detect_duration(question, kw)
                if not found or key not in inputs or inputs[key] is None:
                    continue
                val, unit = found
                if unit == d_per or not math.isclose(float(inputs[key]), val, rel_tol=1e-9):
                    continue
                f = self._DAYS_PER_PERIOD[unit] / self._DAYS_PER_PERIOD[d_per]
                inputs[key] = val * f
                notes.append(f"{key} stated as {val:g} {unit}(s); demand is per {d_per}: {val:g} x {f:.6g} = {val * f:.4f} {d_per}s.")
                if key == "lead_time" and inputs.get("std_lead_time"):
                    s = float(inputs["std_lead_time"])
                    inputs["std_lead_time"] = s * f
                    notes.append(f"std_lead_time {s:g} {unit}(s) -> {s * f:.4f} {d_per}s (same unit as lead time).")
            return inputs, notes

        return inputs, notes

    def _extract_discount_schedule(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Attempt to extract structured discount schedule tiers from text."""
        raw_tiers: List[Tuple[float, Optional[float], float]] = []

        # Pattern 1: Bounded ranges, e.g.:
        # "0-499: $30", "1 to 499 units at $50", "Tier 1: 0 - 499 for $20"
        range_matches = re.findall(
            r"(?:tier\s*\d+[:\s]*)?([\d,]+)\s*(?:-|to)\s*([\d,]+)\s*(?:units)?\s*[:@=]?\s*(?:at|for)?\s*\$?\s*([\d\.]+)",
            text,
            re.IGNORECASE,
        )
        for m in range_matches:
            min_q = float(m[0].replace(",", ""))
            max_q = float(m[1].replace(",", ""))
            price = float(m[2].replace("$", "").rstrip("."))
            raw_tiers.append((min_q, max_q, price))

        # Pattern 2: Open-ended upper tiers, e.g.:
        # "1000+: $25", "1000 or more: $25", "1000 and above at $20", "1000+ units @ $40"
        plus_matches = re.findall(
            r"(?:tier\s*\d+[:\s]*)?([\d,]+)\s*(?:\+|or\s+more|and\s+above)\s*(?:units)?\s*[:@=]?\s*(?:at|for)?\s*\$?\s*([\d\.]+)",
            text,
            re.IGNORECASE,
        )
        for m in plus_matches:
            min_q = float(m[0].replace(",", ""))
            price = float(m[1].replace("$", "").rstrip("."))
            raw_tiers.append((min_q, None, price))

        if not raw_tiers:
            # Fallback Pattern 3: "1 to 499: $50"
            alt_matches = re.findall(
                r"([\d,]+)\s*(?:to|-)\s*([\d,]+|\+)?\s*(?:units)?\s*[:@=]?\s*\$?\s*([\d\.]+)",
                text,
                re.IGNORECASE,
            )
            for m in alt_matches:
                min_q = float(m[0].replace(",", ""))
                max_str = m[1].replace(",", "").replace("+", "").strip() if m[1] else ""
                max_q = float(max_str) if max_str and max_str != "+" else None
                price = float(m[2].replace("$", "").rstrip("."))
                raw_tiers.append((min_q, max_q, price))

        if not raw_tiers:
            return None

        # Deduplicate and sort by min_qty
        seen = set()
        unique_tiers = []
        for min_q, max_q, p in raw_tiers:
            key = (min_q, max_q, p)
            if key not in seen:
                seen.add(key)
                unique_tiers.append((min_q, max_q, p))

        unique_tiers.sort(key=lambda x: x[0])

        schedule = []
        for idx, (min_q, max_q, p) in enumerate(unique_tiers):
            schedule.append({
                "tier": idx + 1,
                "min_qty": min_q,
                "max_qty": max_q,
                "price": p,
            })

        return schedule

    # ------------------------------------------------------------------------
    # STEP 3: SELECT PYTHON TOOL
    # ------------------------------------------------------------------------
    def select_tool(self, model: str) -> str:
        """Select the appropriate Python inventory tool for the identified model."""
        if model not in self.TOOL_MAPPING:
            raise ValueError(f"Unknown inventory model '{model}'. Cannot select tool.")
        return self.TOOL_MAPPING[model]

    # ------------------------------------------------------------------------
    # STEP 4: CALCULATE (STRICT PYTHON EXECUTION)
    # ------------------------------------------------------------------------
    def calculate(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected Python tool with the extracted inputs."""
        tool_func = getattr(inventory_tools, tool_name, None)
        if tool_func is None:
            raise AttributeError(f"Tool function '{tool_name}' not found in inventory_tools.")

        # Execute pure Python tool
        return tool_func(**inputs)

    # ------------------------------------------------------------------------
    # STEP 5: VERIFY
    # ------------------------------------------------------------------------
    def verify(self, model: str, calculation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Verify mathematical integrity, convexity, economic bounds, and policy logic."""
        verification_data = calculation_results.get("verification", {})
        audit_checks = []

        if model == self.MODEL_EOQ:
            holding_eq_ordering = verification_data.get("holding_equals_ordering", False)
            convexity = verification_data.get("convexity_verified", False)
            audit_checks.append({
                "check": "First-Order Condition (Holding Cost == Ordering Cost at Q*)",
                "passed": holding_eq_ordering,
                "detail": f"Discrepancy: ${verification_data.get('holding_ordering_discrepancy', 0.0):.6f}",
            })
            audit_checks.append({
                "check": "Second-Order Convexity (d2TC/dQ2 > 0)",
                "passed": convexity,
                "detail": f"d2TC/dQ2 = {verification_data.get('second_derivative_value', 0.0):.8f}",
            })

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            cost_min = verification_data.get("cost_minimality_verified", False)
            boundary = verification_data.get("boundary_verified", False)
            audit_checks.append({
                "check": "Cost Minimality (Optimal candidate strictly minimizes total annual cost)",
                "passed": cost_min,
                "detail": "Verified across all feasible and price-break candidate points.",
            })
            audit_checks.append({
                "check": "Tier Quantity Boundary Adherence",
                "passed": boundary,
                "detail": "Selected order quantity respects the minimum and maximum tier constraints.",
            })

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            csl_valid = verification_data.get("service_level_valid", False)
            ss_valid = verification_data.get("ss_non_negative", False)
            rop_valid = verification_data.get("rop_ge_lead_time_demand", False)
            audit_checks.append({
                "check": "Service Level Boundary (0 < CSL < 1)",
                "passed": csl_valid,
                "detail": f"Cycle service level is valid.",
            })
            audit_checks.append({
                "check": "Safety Stock Non-Negativity (SS >= 0)",
                "passed": ss_valid,
                "detail": "Buffer stock is non-negative.",
            })
            audit_checks.append({
                "check": "Reorder Point Dominance (ROP >= Expected Lead Time Demand)",
                "passed": rop_valid,
                "detail": "Reorder point adequately covers baseline replenishment period demand.",
            })

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            prot_valid = bool(verification_data.get("protection_period_valid") or verification_data.get("protection_interval_valid", False))
            target_valid = bool(verification_data.get("target_ge_expected_demand") or verification_data.get("target_level_ge_expected_demand", False))
            q_valid = bool(verification_data.get("order_quantity_non_negative", False))
            audit_checks.append({
                "check": "Protection Period Positivity (P + L > 0)",
                "passed": prot_valid,
                "detail": "Protection interval covers review duration plus replenishment lag.",
            })
            audit_checks.append({
                "check": "Base Stock Level Sufficiency (M >= Protection Demand)",
                "passed": target_valid,
                "detail": "Target level covers total expected demand over (P + L).",
            })
            audit_checks.append({
                "check": "Non-Negative Order Sizing (Q >= 0)",
                "passed": q_valid,
                "detail": "Order quantity correctly respects existing inventory position.",
            })

        elif model == self.MODEL_NEWSVENDOR:
            cr_valid = bool(verification_data.get("critical_ratio_valid", False))
            margins_valid = bool(verification_data.get("margins_positive", False))
            q_non_neg = bool(verification_data.get("q_non_negative", False))
            audit_checks.append({
                "check": "Critical Ratio Valid Range (0 < CR < 1)",
                "passed": cr_valid,
                "detail": "Service probability fractile is bounded in unit interval.",
            })
            audit_checks.append({
                "check": "Underage & Overage Margins Positive (Cu > 0, Co > 0)",
                "passed": margins_valid,
                "detail": "Selling price exceeds cost, and cost exceeds salvage value.",
            })
            audit_checks.append({
                "check": "Order Quantity Non-Negative (Q* >= 0)",
                "passed": q_non_neg,
                "detail": "Recommended order quantity is feasible and non-negative.",
            })

        # Fold in the tool's own worksheet checks (computed deterministically in Python)
        for c in calculation_results.get("verification_checks", []):
            audit_checks.append({"check": c["check"], "passed": c["passed"],
                                 "detail": f"expected={c['expected']}, actual={c['actual']}"})

        all_passed = all(c["passed"] for c in audit_checks)

        return {
            "status": "PASSED" if all_passed else "WARNING",
            "audit_checks": audit_checks,
            "notes": verification_data.get("notes", "Mathematical checks verified."),
        }

    # ------------------------------------------------------------------------
    # STEP 6: EXPLAIN
    # ------------------------------------------------------------------------
    def explain(
        self,
        model: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> str:
        """Generate clear, professional, executive-ready explanation and managerial insights."""
        lines = []
        inp = results.get("inputs", inputs)

        if model == self.MODEL_EOQ:
            r = results["results"]
            lines.append("### 🎯 Executive Decision & Inventory Policy")
            lines.append(f"- **Recommended Order Quantity (EOQ):** **{r['eoq_integer']:,} units** (exact: {r['eoq']:.2f} units)")
            lines.append(f"- **Annual Order Frequency:** **{r['orders_per_year']:.1f} orders/year**")
            lines.append(f"- **Cycle Order Interval:** Every **{r['cycle_time_days']:.1f} days**")
            if r.get("reorder_point") is not None:
                lines.append(f"- **Reorder Point (ROP):** Reorder when inventory reaches **{r['reorder_point']:.1f} units**")

            lines.append("\n### 💰 Cost Breakdown")
            lines.append(f"- **Annual Ordering Cost:** ${r['annual_ordering_cost']:,.2f}")
            lines.append(f"- **Annual Holding Cost:** ${r['annual_holding_cost']:,.2f}")
            lines.append(f"- **Total Annual Inventory Policy Cost:** **${r['total_policy_cost']:,.2f}**")
            if r.get("total_purchase_cost") is not None:
                lines.append(f"- **Annual Material Purchase Cost:** ${r['total_purchase_cost']:,.2f}")
                lines.append(f"- **Total Annual Cost (Acquisition + Policy):** **${r['total_annual_cost']:,.2f}**")

            lines.append("\n### 🧮 Mathematical Calculation & Formula")
            d_val = f"{inp.get('annual_demand', 0):,}"
            s_val = f"{inp.get('order_cost', 0):,.2f}"
            h_val = f"{inp.get('holding_cost_per_unit', 0):.2f}"
            eoq_val = f"{r['eoq']:.2f}"
            formula_md = (
                f"$$\\text{{EOQ}} = \\sqrt{{\\frac{{2 \\cdot D \\cdot S}}{{H}}}} = "
                f"\\sqrt{{\\frac{{2 \\times {d_val} \\times {s_val}}}{{{h_val}}}}} = "
                f"{eoq_val} \\text{{ units}}$$"
            )
            lines.append(formula_md)

            lines.append("\n### 🔍 Verification & Sanity Audit")
            lines.append(f"- **Verification Status:** `{verification['status']}`")
            lines.append(f"- {verification['notes']}")

            lines.append("\n### 💡 Managerial & Operational Insights")
            lines.append("- **Holding vs Ordering Trade-Off:** At optimal EOQ, holding and ordering costs are perfectly balanced. Ordering in larger batches will inflate carrying costs, while smaller batches will escalate ordering overhead.")
            lines.append("- **Robustness Property:** Total cost curve is flat around EOQ (square-root sensitivity). A 20% deviation in batch size only increases total inventory policy cost by approximately 1.6%.")

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            opt = results["optimal_decision"]
            lines.append("### 🎯 Executive Decision & Inventory Policy")
            lines.append(f"- **Optimal Order Quantity:** **{opt['optimal_order_quantity']:,} units** in **Tier {opt['tier']}**")
            lines.append(f"- **Negotiated Unit Price:** **${opt['unit_price']:.2f} per unit**")
            lines.append(f"- **Total Annual Cost:** **${opt['total_annual_cost']:,.2f}**")

            lines.append("\n### 📊 Tier-by-Tier Evaluation Summary")
            for t in results["tier_evaluations"]:
                max_str = f"{t['max_qty']:,}" if t["max_qty"] is not None else "Unlimited"
                is_opt = " ⭐ OPTIMAL" if t["tier"] == opt["tier"] else ""
                lines.append(f"- **Tier {t['tier']}** [{t['min_qty']:,} to {max_str} units @ ${t['unit_price']:.2f}]: "
                             f"Candidate Q = {t['candidate_qty']} ({t['selection_logic']}) → Total Cost: ${t.get('total_annual_cost', 0):,.2f}{is_opt}")

            lines.append("\n### 🔍 Verification & Sanity Audit")
            lines.append(f"- **Verification Status:** `{verification['status']}`")
            lines.append(f"- {verification['notes']}")

            lines.append("\n### 💡 Managerial & Operational Insights")
            lines.append("- **Volume Discount Trade-Off:** Taking advantage of price breaks often requires ordering larger quantities than the basic EOQ. The financial analysis proves that the annual purchase savings and reduced ordering frequency substantially exceed the additional working capital and carrying cost.")

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            r = results["results"]
            csl_val = inp.get("service_level", 0.95)
            z_val = inp.get("z_score", 1.645)
            csl_pct = csl_val * 100.0
            lines.append("### 🎯 Executive Decision & Inventory Policy")
            lines.append(f"- **Target Service Level:** **{csl_pct:.1f}% Cycle Service Level** (z = {z_val:.3f})")
            lines.append(f"- **Recommended Safety Stock:** **{r['safety_stock_integer']:,} units** (exact: {r['safety_stock']:.2f})")
            lines.append(f"- **Reorder Point (ROP):** **{r['reorder_point_integer']:,} units** (exact: {r['reorder_point']:.2f})")
            lines.append(f"- **Action Trigger:** Place a replenishment order whenever inventory drops to or below **{r['reorder_point_integer']:,} units**.")

            lines.append("\n### 🧮 Lead Time Demand & Uncertainty Breakdown")
            lines.append(f"- **Expected Demand During Lead Time (d × L):** {r['expected_lead_time_demand']:,.2f} units")
            lines.append(f"- **Lead Time Demand Standard Deviation (σ_DL):** {r['lead_time_demand_std']:.2f} units")
            if r.get("annual_safety_stock_holding_cost") is not None:
                lines.append(f"- **Annual Safety Stock Holding Cost:** ${r['annual_safety_stock_holding_cost']:,.2f}")

            lines.append("\n### 🔍 Verification & Sanity Audit")
            lines.append(f"- **Verification Status:** `{verification['status']}`")
            lines.append(f"- {verification['notes']}")

            lines.append("\n### 💡 Managerial & Operational Insights")
            lines.append(f"- **Buffer Protection:** Safety stock protects against stockouts during replenishment lag. Under this policy, {csl_pct:.1f}% of replenishment cycles will experience zero stockouts.")
            lines.append("- **Lead Time Reduction Impact:** Because safety stock scales with the square root of lead time, reducing supplier lead time or demand variance is the fastest lever to decrease working capital.")

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            r = results["results"]
            lines.append("### 🎯 Executive Decision & Inventory Policy")
            lines.append(f"- **Target Order-Up-To Level (Base Stock):** **{r['target_level_integer']:,} units** (exact: {r['target_level']:.2f})")
            lines.append(f"- **Recommended Order Quantity Right Now:** **{r['order_quantity_integer']:,} units** (exact: {r['order_quantity']:.2f})")
            eff_pos = inp.get("effective_inventory_position", 0.0)
            cur_inv = inp.get("current_inventory", 0.0)
            on_ord = inp.get("on_order", 0.0)
            back_ord = inp.get("backorders", 0.0)
            lines.append(f"- **Current Effective Inventory Position:** {eff_pos:,.1f} units "
                         f"(On-hand: {cur_inv:,.0f}, On-order: {on_ord:,.0f}, Backorders: {back_ord:,.0f})")

            lines.append("\n### 🧮 Protection Interval Dynamics")
            lines.append(f"- **Review Period (P):** {inp.get('review_period')} periods")
            lines.append(f"- **Lead Time (L):** {inp.get('lead_time')} periods")
            lines.append(f"- **Total Protection Interval (P + L):** {inp.get('protection_interval')} periods")
            lines.append(f"- **Expected Demand Over (P + L):** {r['expected_protection_demand']:,} units")
            lines.append(f"- **Safety Stock Component:** {r['safety_stock']:.2f} units")

            lines.append("\n### 🔍 Verification & Sanity Audit")
            lines.append(f"- **Verification Status:** `{verification['status']}`")
            lines.append(f"- {verification['notes']}")

            lines.append("\n### 💡 Managerial & Operational Insights")
            lines.append("- **Periodic Coordination:** Periodic review synchronizes orders across multiple products from the same vendor, reducing shipping overhead, but requires higher safety stock because the protection interval covers both review frequency and lead time (P + L).")

        elif model == self.MODEL_NEWSVENDOR:
            r = results["results"]
            lines.append("### 🎯 Executive Decision & Optimal Order Policy")
            lines.append(f"- **Optimal Order Quantity (Q*):** **{r['optimal_order_quantity_integer']:,} units** (continuous: {r['optimal_order_quantity']:.2f})")
            lines.append(f"- **Critical Ratio (Target In-Stock Probability):** **{r['critical_ratio'] * 100:.2f}%**")
            lines.append(f"- **Expected Profit:** **${r['expected_profit']:,.2f}**")

            lines.append("\n### ⚖️ Marginal Economic Costs")
            lines.append(f"- **Cost of Underage (Cu - Lost Margin per Unit):** ${r['cost_of_underage_cu']:,.2f}")
            lines.append(f"- **Cost of Overage (Co - Loss per Excess Unit):** ${r['cost_of_overage_co']:,.2f}")
            lines.append(f"- **Critical Ratio Formula:** Cu / (Cu + Co) = {r['critical_ratio']:.4f}")

            if "expected_profit_perfect_info" in r:
                lines.append("\n### 📈 Financial Performance & Risk Metrics")
                lines.append(f"- **Expected Sales:** {r['expected_sales']:.1f} units")
                lines.append(f"- **Expected Leftovers (Overstocking):** {r['expected_leftover_units']:.1f} units")
                lines.append(f"- **Expected Lost Sales (Understocking):** {r['expected_lost_sales']:.1f} units")
                lines.append(f"- **Expected Revenue:** ${r['expected_revenue']:,.2f}")
                lines.append(f"- **Expected Salvage Recovery:** ${r['expected_salvage_recovery']:,.2f}")
                lines.append(f"- **Total Purchasing Investment:** ${r['total_purchasing_cost']:,.2f}")
                lines.append(f"- **Value of Perfect Information (EVPI):** ${r['expected_value_of_perfect_info_evpi']:,.2f}")

            lines.append("\n### 🔍 Verification & Sanity Audit")
            lines.append(f"- **Verification Status:** `{verification['status']}`")
            lines.append(f"- {verification['notes']}")

            lines.append("\n### 💡 Managerial & Operational Insights")
            if r["cost_of_underage_cu"] > r["cost_of_overage_co"]:
                lines.append(f"- **Aggressive Stocking Rationale:** Because unit profit margin (${r['cost_of_underage_cu']:.2f}) exceeds overstock loss (${r['cost_of_overage_co']:.2f}), the critical ratio ({r['critical_ratio']:.2f}) is > 50%. The optimal decision orders **more** than average demand ({inp.get('demand_mean', 0):,}) to capture high-margin upside.")
            else:
                lines.append(f"- **Conservative Stocking Rationale:** Because the loss on unsold units (${r['cost_of_overage_co']:.2f}) exceeds the profit margin (${r['cost_of_underage_cu']:.2f}), the optimal policy orders **less** than average demand to protect against clearance losses.")

        lines.append("")
        lines.append(self.render_worksheet(results))
        return "\n".join(lines)

    # ------------------------------------------------------------------------
    # STUDENT-VERIFIABLE WORKSHEET (8 sections, rendered verbatim from the tool)
    # ------------------------------------------------------------------------
    @staticmethod
    def _fmt_value(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:,.4f}".rstrip("0").rstrip(".")
        return str(v)

    def render_worksheet(self, results: Dict[str, Any]) -> str:
        """Render the 8-part verifiable worksheet.

        Every number shown is copied from the deterministic tool's output; this method
        only formats text and never computes or alters numerical values.
        """
        if "formula" not in results:
            return ""
        fv = self._fmt_value
        out: List[str] = ["---", "## 📘 Verifiable Worksheet (all numbers produced by the Python tool)"]

        out.append("\n### 1. Inputs Extracted")
        for note in results.get("unit_conversions", []):
            out.append(f"- **Unit conversion:** {note}")
        for k, v in results.get("inputs_extracted", {}).items():
            if isinstance(v, list):
                out.append(f"- `{k}`:")
                for item in v:
                    out.append(f"  - {item}")
            else:
                out.append(f"- `{k}` = {fv(v)}")

        out.append("\n### 2. Variable Definitions")
        for k, v in results.get("variable_definitions", {}).items():
            out.append(f"- **{k}**: {v}")

        out.append("\n### 3. Formula")
        for f in results["formula"]:
            out.append(f"- `{f}`")

        out.append("\n### 4. Formula with Substituted Values")
        for f in results.get("formula_substituted", []):
            out.append(f"- `{f}`")

        out.append("\n### 5. Intermediate Calculations")
        for k, v in results.get("intermediate_calculations", {}).items():
            if isinstance(v, list):
                out.append(f"- **{k}**:")
                for item in v:
                    if isinstance(item, dict):
                        out.append("  - " + ", ".join(f"{ik}={fv(iv)}" for ik, iv in item.items() if iv is not None))
                    else:
                        out.append(f"  - {item}")
            else:
                out.append(f"- **{k}**: {fv(v)}")

        out.append("\n### 6. Final Answer")
        for k, v in results.get("final_answer", {}).items():
            out.append(f"- **{k}**: {fv(v)}")

        out.append("\n### 7. Verification Checks")
        out.append("| Check | Expected | Actual | Result |")
        out.append("|---|---|---|---|")
        for c in results.get("verification_checks", []):
            mark = "✅ PASS" if c["passed"] else "❌ FAIL"
            out.append(f"| {c['check']} | {fv(c['expected'])} | {fv(c['actual'])} | {mark} |")

        out.append("\n### 8. Assumptions")
        for a in results.get("assumptions", []):
            out.append(f"- {a}")

        return "\n".join(out)

    # ------------------------------------------------------------------------
    # FULL END-TO-END WORKFLOW PIPELINE
    # ------------------------------------------------------------------------
    def process_question(self, question: str) -> InventoryDecisionResult:
        """Execute the full 7-stage inventory decision architecture pipeline.

        Pipeline:
        Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        """
        trace: List[AgentStep] = []
        gemini_api_key = get_gemini_api_key()
        extraction_method = "Gemini API (gemini-2.5-flash)" if gemini_api_key else "Deterministic NLP & Regex Extractor"

        # Step 0: Question Received
        trace.append(AgentStep(
            step_name="Customer Question Received",
            status="SUCCESS",
            details={
                "raw_question": question,
                "api_key_detected": bool(gemini_api_key),
                "active_extraction_engine": extraction_method,
            },
            description=f"Captured natural language inventory inquiry. Extraction Engine: {extraction_method}.",
        ))

        model: str = self.MODEL_EOQ
        confidence: float = 0.5
        extracted_inputs: Dict[str, Any] = {}
        missing_inputs: List[str] = []

        # Attempt Gemini API extraction if key is configured
        gemini_success = False
        if gemini_api_key:
            try:
                gemini_res = self.extract_with_gemini(question, gemini_api_key)
                if gemini_res:
                    model, confidence, extracted_inputs, missing_inputs = gemini_res
                    gemini_success = True
                    trace.append(AgentStep(
                        step_name="Gemini API Model & Input Extraction",
                        status="SUCCESS",
                        details={
                            "model": model,
                            "confidence": confidence,
                            "inputs_count": len(extracted_inputs),
                            "missing_count": len(missing_inputs),
                        },
                        description=f"Gemini API classified model as {model} ({confidence * 100:.1f}%) and extracted {len(extracted_inputs)} parameters.",
                    ))
            except Exception as e:
                trace.append(AgentStep(
                    step_name="Gemini API Call",
                    status="WARNING",
                    details={"error": str(e)},
                    description="Gemini API encountered error; falling back to deterministic regex extractor.",
                ))
                gemini_success = False

        # Fallback to deterministic NLP / Regex extractor if Gemini unavailable or returned partial
        if not gemini_success:
            extraction_method = "Deterministic NLP & Regex Extractor"
            # Step 1: Identify Model
            try:
                model, confidence, all_scores = self.identify_model(question)
                trace.append(AgentStep(
                    step_name="Identify Model",
                    status="SUCCESS",
                    details={
                        "identified_model": model,
                        "model_display_name": self.MODEL_DISPLAY_NAMES.get(model, model),
                        "confidence": confidence,
                        "candidate_scores": all_scores,
                    },
                    description=f"Identified decision model as {self.MODEL_DISPLAY_NAMES.get(model, model)} with confidence {confidence * 100:.1f}%.",
                ))
            except Exception as e:
                trace.append(AgentStep(
                    step_name="Identify Model",
                    status="FAILED",
                    details={"error": str(e)},
                    description="Failed to classify inventory model.",
                ))
                return InventoryDecisionResult(
                    customer_question=question,
                    identified_model="UNKNOWN",
                    model_display_name="Unknown Model",
                    model_confidence=0.0,
                    extracted_inputs={},
                    missing_inputs=[],
                    selected_tool="None",
                    calculation_results=None,
                    verification=None,
                    explanation=f"Error identifying inventory model: {e}",
                    execution_trace=trace,
                    success=False,
                    error_message=str(e),
                    extraction_method=extraction_method,
                )

            # Step 2: Extract Inputs (Deterministic)
            try:
                extracted_inputs, missing_inputs = self.extract_inputs(question, model)
                trace.append(AgentStep(
                    step_name="Extract Inputs",
                    status="SUCCESS" if not missing_inputs else "PARTIAL",
                    details={
                        "extracted_inputs": extracted_inputs,
                        "missing_inputs": missing_inputs,
                    },
                    description=(
                        f"Extracted {len(extracted_inputs)} parameters with full synonym mapping."
                        + (f" Missing required inputs: {', '.join(missing_inputs)}." if missing_inputs else " All required inputs found.")
                    ),
                ))
            except Exception as e:
                trace.append(AgentStep(
                    step_name="Extract Inputs",
                    status="FAILED",
                    details={"error": str(e)},
                    description="Error parsing numeric parameters from question.",
                ))
                return InventoryDecisionResult(
                    customer_question=question,
                    identified_model=model,
                    model_display_name=self.MODEL_DISPLAY_NAMES.get(model, model),
                    model_confidence=confidence,
                    extracted_inputs={},
                    missing_inputs=[],
                    selected_tool="None",
                    calculation_results=None,
                    verification=None,
                    explanation=f"Failed to parse inputs from question: {e}",
                    execution_trace=trace,
                    success=False,
                    error_message=str(e),
                    extraction_method=extraction_method,
                )

        # Step 2b: Normalize time units (deterministic Python, applies to Gemini and regex paths)
        extracted_inputs, unit_notes = self.normalize_time_units(question, model, extracted_inputs)
        if unit_notes:
            trace.append(AgentStep(
                step_name="Normalize Time Units",
                status="SUCCESS",
                details={"conversions": unit_notes, "normalized_inputs": extracted_inputs},
                description="Converted rates to a consistent time basis: " + " ".join(unit_notes),
            ))

        # Step 3: Select Python Tool
        try:
            tool_name = self.select_tool(model)
            trace.append(AgentStep(
                step_name="Select Python Tool",
                status="SUCCESS",
                details={
                    "tool_name": tool_name,
                    "module": "inventory_tools",
                },
                description=f"Selected deterministic Python tool 'inventory_tools.{tool_name}'.",
            ))
        except Exception as e:
            trace.append(AgentStep(
                step_name="Select Python Tool",
                status="FAILED",
                details={"error": str(e)},
                description="Failed to map model to Python tool.",
            ))
            return InventoryDecisionResult(
                customer_question=question,
                identified_model=model,
                model_display_name=self.MODEL_DISPLAY_NAMES.get(model, model),
                model_confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing_inputs,
                selected_tool="None",
                calculation_results=None,
                verification=None,
                explanation=f"Error selecting tool: {e}",
                execution_trace=trace,
                success=False,
                error_message=str(e),
                extraction_method=extraction_method,
            )

        # Handle missing parameters gracefully without guessing or inventing
        if missing_inputs:
            explanation = (
                f"### ⚠️ Clarification Needed for {self.MODEL_DISPLAY_NAMES.get(model, model)}\n\n"
                f"I identified that your question pertains to **{self.MODEL_DISPLAY_NAMES.get(model, model)}**, "
                f"but I need the following additional parameter(s) to compute the exact numerical decision:\n\n"
                + "\n".join([f"- **{m}**" for m in missing_inputs])
                + "\n\n**Currently Extracted Parameters:**\n"
                + json.dumps(extracted_inputs, indent=2)
                + "\n\nPlease provide the missing values so I can execute the calculation without inventing numerical values."
            )
            return InventoryDecisionResult(
                customer_question=question,
                identified_model=model,
                model_display_name=self.MODEL_DISPLAY_NAMES.get(model, model),
                model_confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing_inputs,
                selected_tool=tool_name,
                calculation_results=None,
                verification=None,
                explanation=explanation,
                execution_trace=trace,
                success=False,
                error_message=f"Missing required parameters: {', '.join(missing_inputs)}",
                extraction_method=extraction_method,
            )

        # Step 4: Calculate (Strict Deterministic Python Engine)
        try:
            calc_results = self.calculate(tool_name, extracted_inputs)
            if unit_notes:
                calc_results["unit_conversions"] = unit_notes
            trace.append(AgentStep(
                step_name="Calculate",
                status="SUCCESS",
                details={"results_keys": list(calc_results.keys())},
                description="Executed deterministic Python analytical calculation engine.",
            ))
        except Exception as e:
            trace.append(AgentStep(
                step_name="Calculate",
                status="FAILED",
                details={"error": str(e)},
                description=f"Calculation error in Python tool: {e}",
            ))
            return InventoryDecisionResult(
                customer_question=question,
                identified_model=model,
                model_display_name=self.MODEL_DISPLAY_NAMES.get(model, model),
                model_confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                selected_tool=tool_name,
                calculation_results=None,
                verification=None,
                explanation=f"Calculation error while running '{tool_name}': {e}",
                execution_trace=trace,
                success=False,
                error_message=str(e),
                extraction_method=extraction_method,
            )

        # Step 5: Verify
        try:
            verification_audit = self.verify(model, calc_results)
            trace.append(AgentStep(
                step_name="Verify",
                status=verification_audit.get("status", "PASSED"),
                details=verification_audit,
                description=f"Domain verification {verification_audit.get('status', 'PASSED')}. Checked {len(verification_audit.get('audit_checks', []))} properties.",
            ))
        except Exception as e:
            trace.append(AgentStep(
                step_name="Verify",
                status="WARNING",
                details={"error": str(e)},
                description="Verification audit warning.",
            ))
            verification_audit = {"status": "WARNING", "notes": f"Verification error: {e}"}

        # Step 6: Explain
        try:
            explanation = self.explain(model, extracted_inputs, calc_results, verification_audit)
            trace.append(AgentStep(
                step_name="Explain",
                status="SUCCESS",
                details={"summary_length": len(explanation)},
                description="Generated executive summary, mathematical proof, and managerial insights.",
            ))
        except Exception as e:
            explanation = f"Calculations completed successfully, but error formatting explanation: {e}"
            trace.append(AgentStep(
                step_name="Explain",
                status="WARNING",
                details={"error": str(e)},
                description="Error generating narrative explanation.",
            ))

        return InventoryDecisionResult(
            customer_question=question,
            identified_model=model,
            model_display_name=self.MODEL_DISPLAY_NAMES.get(model, model),
            model_confidence=confidence,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            selected_tool=tool_name,
            calculation_results=calc_results,
            verification=verification_audit,
            explanation=explanation,
            execution_trace=trace,
            success=True,
            error_message=None,
            extraction_method=extraction_method,
        )
