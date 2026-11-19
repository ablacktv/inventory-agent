"""Supply Chain Analytics Deterministic Python Inventory Tools.

This module provides deterministic Python calculation engines for five foundational
inventory decision models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Each tool:
- Validates all inputs strictly
- Never invents missing numerical inputs
- Returns structured results
- Returns intermediate calculations
- Returns the final answer
- Explicitly identifies operational and economic assumptions
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats


# ============================================================================
# SHARED: STUDENT-VERIFIABLE WORKSHEET HELPERS
# ============================================================================

def _check(name: str, expected: Any, actual: Any, tol: float = 1e-6, kind: str = "equal") -> Dict[str, Any]:
    """Build one verification-check record (computed by the tool, never by the LLM)."""
    if kind == "equal":
        passed = abs(float(expected) - float(actual)) <= tol * max(1.0, abs(float(expected)))
    elif kind == "gt":      # actual must be strictly greater than expected
        passed = float(actual) > float(expected)
    elif kind == "ge":      # actual must be >= expected
        passed = float(actual) >= float(expected) - tol
    elif kind == "bool":
        passed = bool(actual)
    else:
        raise ValueError(f"Unknown check kind: {kind}")
    return {
        "check": name,
        "expected": expected,
        "actual": actual,
        "passed": bool(passed),
    }


def _attach_worksheet(
    result: Dict[str, Any],
    variable_definitions: Dict[str, str],
    formulas: List[str],
    formulas_substituted: List[str],
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach the 8-part verifiable worksheet to a tool result.

    Sections: inputs_extracted, variable_definitions, formula, formula_substituted,
    intermediate_calculations (already present), final_answer (already present),
    verification_checks, assumptions (already present).
    """
    result["inputs_extracted"] = {k: v for k, v in result.get("inputs", {}).items() if v is not None}
    result["variable_definitions"] = variable_definitions
    result["formula"] = formulas
    result["formula_substituted"] = formulas_substituted
    result["verification_checks"] = checks
    all_passed = all(c["passed"] for c in checks)
    result["verification"]["checks_passed"] = all_passed
    if not all_passed:
        result["verification"]["status"] = "WARNING"
    result["worksheet_sections"] = [
        "inputs_extracted", "variable_definitions", "formula", "formula_substituted",
        "intermediate_calculations", "final_answer", "verification_checks", "assumptions",
    ]
    return result


# ============================================================================
# 1. ECONOMIC ORDER QUANTITY (EOQ)
# ============================================================================

def calculate_eoq(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    # Aliases for flexible integration
    annual_demand: Optional[float] = None,
    holding_cost_per_unit: Optional[float] = None,
    holding_cost_pct: Optional[float] = None,
    unit_price: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    work_days_per_year: float = 365.0,
) -> Dict[str, Any]:
    """Calculate the Economic Order Quantity (EOQ) and inventory cost breakdown.

    Formula:
        Q* = sqrt((2 * D * S) / H)

    Args:
        demand: Annual demand in units (D > 0). Can also pass as annual_demand.
        order_cost: Fixed setup or ordering cost per order (S > 0).
        holding_cost: Annual holding cost per unit per year (H > 0).
        holding_rate: Annual holding cost as a decimal percentage of unit cost (e.g. 0.20 for 20%).
        unit_cost: Purchase price per unit (C > 0).
        annual_demand: Alias for demand.
        holding_cost_per_unit: Alias for holding_cost.
        holding_cost_pct: Alias for holding_rate.
        unit_price: Alias for unit_cost.
        lead_time_days: Replenishment lead time in days (optional).
        work_days_per_year: Operating work days per year (default 365.0).

    Returns:
        Structured dictionary containing final_answer, intermediate_calculations,
        assumptions, inputs, results, and verification audit.
    """
    # 1. Resolve aliases
    d = demand if demand is not None else annual_demand
    s = order_cost
    h = holding_cost if holding_cost is not None else holding_cost_per_unit
    h_rate = holding_rate if holding_rate is not None else holding_cost_pct
    c = unit_cost if unit_cost is not None else unit_price

    # 2. Strict Input Validation (Never invent missing inputs)
    if d is None:
        raise ValueError("Missing required input: 'demand' (annual demand D). Missing inputs are not invented.")
    if s is None:
        raise ValueError("Missing required input: 'order_cost' (ordering cost per order S). Missing inputs are not invented.")
    if d <= 0:
        raise ValueError(f"Annual demand must be strictly positive; got {d}")
    if s <= 0:
        raise ValueError(f"Order cost must be strictly positive; got {s}")

    # Determine unit annual holding cost H
    if h is not None:
        if h <= 0:
            raise ValueError(f"Holding cost must be strictly positive; got {h}")
        final_h = float(h)
    elif h_rate is not None and c is not None:
        if h_rate <= 0:
            raise ValueError(f"Holding rate percentage must be strictly positive; got {h_rate}")
        if c <= 0:
            raise ValueError(f"Unit cost must be strictly positive; got {c}")
        final_h = float(h_rate * c)
    else:
        raise ValueError(
            "Missing holding cost specification: must provide either 'holding_cost' (>0) "
            "or both 'holding_rate' (>0) and 'unit_cost' (>0). Missing numerical inputs are not invented."
        )

    # 3. Intermediate Calculations
    numerator = 2.0 * d * s
    radicand = numerator / final_h
    eoq_exact = math.sqrt(radicand)
    eoq_rounded = round(eoq_exact)

    orders_per_year = d / eoq_exact
    cycle_time_years = eoq_exact / d
    cycle_time_days = cycle_time_years * work_days_per_year

    annual_ordering_cost = (d / eoq_exact) * s
    annual_holding_cost = (eoq_exact / 2.0) * final_h
    total_policy_cost = annual_ordering_cost + annual_holding_cost

    total_purchase_cost = (d * c) if c is not None else None
    total_annual_cost = (total_purchase_cost + total_policy_cost) if total_purchase_cost is not None else total_policy_cost

    daily_demand = d / work_days_per_year
    reorder_point = (daily_demand * lead_time_days) if lead_time_days is not None else None

    # Verification checks: holding cost == ordering cost at Q*
    cost_discrepancy = abs(annual_ordering_cost - annual_holding_cost)
    holding_equals_ordering = cost_discrepancy < 1e-4
    second_derivative = (2.0 * d * s) / (eoq_exact ** 3)
    convexity_verified = second_derivative > 0

    intermediate = {
        "holding_cost_per_unit": round(final_h, 4),
        "numerator_2DS": round(numerator, 2),
        "radicand_2DS_over_H": round(radicand, 4),
        "exact_eoq": round(eoq_exact, 4),
        "orders_per_year": round(orders_per_year, 2),
        "cycle_time_days": round(cycle_time_days, 2),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_policy_cost": round(total_policy_cost, 2),
        "total_purchase_cost": round(total_purchase_cost, 2) if total_purchase_cost is not None else None,
        "second_derivative_d2TC_dQ2": round(second_derivative, 8),
    }

    final_answer = {
        "order_quantity": round(eoq_exact, 2),
        "order_quantity_integer": eoq_rounded,
        "total_annual_policy_cost": round(total_policy_cost, 2),
        "total_annual_cost": round(total_annual_cost, 2),
        "orders_per_year": round(orders_per_year, 2),
        "order_cycle_days": round(cycle_time_days, 1),
    }
    if reorder_point is not None:
        final_answer["reorder_point"] = round(reorder_point, 2)

    assumptions = [
        "Continuous review inventory control system",
        "Deterministic, constant, and continuous demand rate over the entire planning horizon",
        "Instantaneous order delivery or constant deterministic supplier replenishment lead time",
        "Fixed ordering / setup cost per order, independent of batch quantity",
        "Constant holding cost per unit per year",
        "No quantity discounts and no inventory shortages (stockouts or backorders) allowed",
    ]

    verification = {
        "status": "PASSED" if (holding_equals_ordering and convexity_verified) else "WARNING",
        "holding_equals_ordering": holding_equals_ordering,
        "holding_ordering_discrepancy": round(cost_discrepancy, 6),
        "convexity_verified": convexity_verified,
        "second_derivative_value": round(second_derivative, 8),
        "notes": (
            f"Verified: Holding cost (${annual_holding_cost:,.2f}) equals ordering cost (${annual_ordering_cost:,.2f}) "
            f"at Q*, with second derivative d2TC/dQ2 = {second_derivative:.6f} > 0 confirming strict global convexity."
        ),
    }

    _result = {
        "model": "Economic Order Quantity (EOQ)",
        "final_answer": final_answer,
        "intermediate_calculations": intermediate,
        "assumptions": assumptions,
        "verification": verification,
        "inputs": {
            "demand": d,
            "annual_demand": d,
            "order_cost": s,
            "holding_cost_per_unit": final_h,
            "unit_cost": c,
            "unit_price": c,
            "lead_time_days": lead_time_days,
            "work_days_per_year": work_days_per_year,
        },
        "results": {
            "eoq": round(eoq_exact, 2),
            "eoq_integer": eoq_rounded,
            "orders_per_year": round(orders_per_year, 2),
            "cycle_time_days": round(cycle_time_days, 1),
            "daily_demand": round(daily_demand, 2),
            "reorder_point": round(reorder_point, 2) if reorder_point is not None else None,
            "annual_ordering_cost": round(annual_ordering_cost, 2),
            "annual_holding_cost": round(annual_holding_cost, 2),
            "total_policy_cost": round(total_policy_cost, 2),
            "total_purchase_cost": round(total_purchase_cost, 2) if total_purchase_cost is not None else None,
            "total_annual_cost": round(total_annual_cost, 2),
        },
    }

    _tc = lambda q: (d / q) * s + (q / 2.0) * final_h
    _from_rate = (h is None and h_rate is not None)
    return _attach_worksheet(
        _result,
        variable_definitions={
            "D": "Annual demand (units/year)",
            "S": "Ordering / setup cost per order ($/order)",
            "H": "Annual holding cost per unit ($/unit/year)" + (" = holding rate h x unit cost C" if _from_rate else ""),
            "Q*": "Economic order quantity (units/order)",
            "TC": "Total annual ordering + holding cost ($/year)",
        },
        formulas=(["H = h x C"] if _from_rate else []) + [
            "Q* = sqrt(2 x D x S / H)",
            "Annual ordering cost = (D / Q*) x S",
            "Annual holding cost = (Q* / 2) x H",
            "TC = (D / Q*) x S + (Q* / 2) x H",
        ],
        formulas_substituted=(
            ([f"H = {h_rate} x {c} = {final_h:.4f}"] if _from_rate else [])
            + [
                f"Q* = sqrt(2 x {d:,.4f} x {s:,.4f} / {final_h:.4f}) = sqrt({radicand:,.4f}) = {eoq_exact:,.4f}",
                f"Annual ordering cost = ({d:,.4f} / {eoq_exact:,.4f}) x {s:,.4f} = {annual_ordering_cost:,.4f}",
                f"Annual holding cost = ({eoq_exact:,.4f} / 2) x {final_h:.4f} = {annual_holding_cost:,.4f}",
                f"TC = {annual_ordering_cost:,.4f} + {annual_holding_cost:,.4f} = {total_policy_cost:,.4f}",
            ]
        ),
        checks=[
            _check("Q*^2 x H equals 2 x D x S", numerator, eoq_exact ** 2 * final_h, tol=1e-9),
            _check("Annual ordering cost equals annual holding cost at Q*", annual_ordering_cost, annual_holding_cost, tol=1e-9),
            _check("Second derivative of TC is positive (minimum)", 0.0, second_derivative, kind="gt"),
            _check("TC at 0.9 x Q* is greater than TC at Q* (expected = TC at Q*)", total_policy_cost, _tc(0.9 * eoq_exact), kind="gt"),
            _check("TC at 1.1 x Q* is greater than TC at Q* (expected = TC at Q*)", total_policy_cost, _tc(1.1 * eoq_exact), kind="gt"),
        ],
    )


# ============================================================================
# 2. QUANTITY DISCOUNT
# ============================================================================

def calculate_quantity_discount(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    tiers: Optional[List[Dict[str, Any]]] = None,
    holding_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    # Aliases
    annual_demand: Optional[float] = None,
    discount_schedule: Optional[List[Dict[str, Any]]] = None,
    fixed_holding_cost: Optional[float] = None,
    holding_cost_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate the optimal order quantity under an all-units quantity discount schedule.

    Evaluates feasible order quantities across all discount tiers and compares
    total annual cost (Purchase Cost + Ordering Cost + Holding Cost).

    Args:
        demand: Annual demand in units (D > 0).
        order_cost: Fixed ordering cost per order (S > 0).
        tiers: List of tier dicts with min_qty, max_qty, and price.
        holding_cost: Fixed annual holding cost per unit (constant across tiers).
        holding_rate: Annual holding cost percentage as fraction of unit price (e.g. 0.20 for 20%).
        annual_demand: Alias for demand.
        discount_schedule: Alias for tiers.
        fixed_holding_cost: Alias for holding_cost.
        holding_cost_pct: Alias for holding_rate.

    Returns:
        Structured dictionary containing final_answer, intermediate_calculations,
        assumptions, inputs, results, and verification audit.
    """
    # 1. Resolve aliases
    d = demand if demand is not None else annual_demand
    s = order_cost
    tier_list = tiers if tiers is not None else discount_schedule
    h_fixed = holding_cost if holding_cost is not None else fixed_holding_cost
    h_pct = holding_rate if holding_rate is not None else holding_cost_pct

    # 2. Strict Input Validation (Never invent missing inputs)
    if d is None:
        raise ValueError("Missing required input: 'demand' (annual demand D). Missing inputs are not invented.")
    if s is None:
        raise ValueError("Missing required input: 'order_cost' (ordering cost S). Missing inputs are not invented.")
    if d <= 0:
        raise ValueError(f"Annual demand must be strictly positive; got {d}")
    if s <= 0:
        raise ValueError(f"Order cost must be strictly positive; got {s}")
    if not tier_list or not isinstance(tier_list, list):
        raise ValueError("Missing required input: 'tiers' must be a non-empty list of discount schedule tiers.")
    if h_pct is None and h_fixed is None:
        raise ValueError(
            "Missing holding cost specification: must provide either 'holding_rate' "
            "(percentage of price) or 'holding_cost' (fixed dollar cost). Missing inputs are not invented."
        )

    # 3. Intermediate Tier Evaluations
    tier_evaluations = []
    valid_candidates = []

    for idx, tier in enumerate(tier_list):
        tier_num = tier.get("tier", idx + 1)
        min_q = float(tier.get("min_qty", 0))
        max_q_raw = tier.get("max_qty")
        max_q = float(max_q_raw) if max_q_raw is not None and max_q_raw != "" else float("inf")
        price = float(tier["price"])

        if price <= 0:
            raise ValueError(f"Tier {tier_num} price must be positive; got {price}")
        if min_q < 0:
            raise ValueError(f"Tier {tier_num} min_qty cannot be negative; got {min_q}")

        # Determine unit holding cost for this tier
        if h_pct is not None:
            if h_pct <= 0:
                raise ValueError(f"Holding rate must be positive; got {h_pct}")
            h_tier = h_pct * price
        else:
            if h_fixed is None or h_fixed <= 0:
                raise ValueError("Fixed holding cost must be strictly positive.")
            h_tier = float(h_fixed)

        # Unconstrained EOQ for this tier
        eoq_raw = math.sqrt((2.0 * d * s) / h_tier)

        # Feasibility classification
        is_eoq_feasible = min_q <= eoq_raw <= max_q
        candidate_q: Optional[float] = None
        candidate_reason = ""

        if is_eoq_feasible:
            candidate_q = eoq_raw
            candidate_reason = "Unconstrained EOQ is feasible within tier range"
        elif eoq_raw < min_q:
            candidate_q = min_q
            candidate_reason = "EOQ is below tier; price break minimum selected"
        else:
            candidate_q = None
            candidate_reason = "EOQ exceeds tier ceiling; superseded by cheaper price tiers"

        tier_record: Dict[str, Any] = {
            "tier": tier_num,
            "min_qty": min_q,
            "max_qty": max_q_raw,
            "unit_price": price,
            "holding_cost_per_unit": round(h_tier, 4),
            "unconstrained_eoq": round(eoq_raw, 2),
            "is_eoq_feasible": is_eoq_feasible,
            "candidate_qty": round(candidate_q, 2) if candidate_q is not None else None,
            "selection_logic": candidate_reason,
        }

        if candidate_q is not None and candidate_q > 0:
            annual_purchase = d * price
            annual_ordering = (d / candidate_q) * s
            annual_holding = (candidate_q / 2.0) * h_tier
            total_annual = annual_purchase + annual_ordering + annual_holding

            tier_record.update({
                "annual_purchase_cost": round(annual_purchase, 2),
                "annual_ordering_cost": round(annual_ordering, 2),
                "annual_holding_cost": round(annual_holding, 2),
                "total_inventory_cost": round(annual_ordering + annual_holding, 2),
                "total_annual_cost": round(total_annual, 2),
            })
            valid_candidates.append(tier_record)

        tier_evaluations.append(tier_record)

    if not valid_candidates:
        raise ValueError("No feasible order quantity candidates could be evaluated for the discount schedule.")

    # 4. Final Answer Selection (Minimizing Total Annual Cost)
    best_candidate = min(valid_candidates, key=lambda x: x["total_annual_cost"])

    final_answer = {
        "optimal_order_quantity": best_candidate["candidate_qty"],
        "optimal_tier": best_candidate["tier"],
        "unit_price": best_candidate["unit_price"],
        "minimum_total_annual_cost": best_candidate["total_annual_cost"],
        "annual_purchase_cost": best_candidate["annual_purchase_cost"],
        "annual_inventory_policy_cost": best_candidate["total_inventory_cost"],
    }

    assumptions = [
        "All-units quantity discount schedule (entire order receives the discounted price)",
        "Constant and continuous demand over the annual planning period",
        "Supplier provides full quantity fulfillment without stockouts or partial backlogs",
        "Holding cost scales with discounted purchase price (when holding_rate is applied) or remains constant per unit",
        "Fixed ordering cost remains constant across all order quantities",
    ]

    all_costs = [c["total_annual_cost"] for c in valid_candidates]
    cost_minimality_verified = abs(best_candidate["total_annual_cost"] - min(all_costs)) < 1e-4

    opt_q = best_candidate["candidate_qty"]
    opt_min = best_candidate["min_qty"]
    opt_max_raw = best_candidate["max_qty"]
    opt_max = float(opt_max_raw) if opt_max_raw is not None else float("inf")
    boundary_verified = (opt_min <= opt_q <= opt_max)

    verification = {
        "status": "PASSED" if (cost_minimality_verified and boundary_verified) else "WARNING",
        "cost_minimality_verified": cost_minimality_verified,
        "boundary_verified": boundary_verified,
        "notes": (
            f"Verified: Order quantity {opt_q} in Tier {best_candidate['tier']} strictly minimizes "
            f"total annual cost at ${best_candidate['total_annual_cost']:,.2f} while satisfying [{opt_min}, {opt_max_raw}]."
        ),
    }

    _result = {
        "model": "Quantity Discount Analysis",
        "final_answer": final_answer,
        "intermediate_calculations": {
            "tier_evaluations": tier_evaluations,
            "candidate_count": len(valid_candidates),
        },
        "assumptions": assumptions,
        "verification": verification,
        "optimal_decision": {
            "tier": best_candidate["tier"],
            "optimal_order_quantity": best_candidate["candidate_qty"],
            "unit_price": best_candidate["unit_price"],
            "annual_purchase_cost": best_candidate["annual_purchase_cost"],
            "annual_ordering_cost": best_candidate["annual_ordering_cost"],
            "annual_holding_cost": best_candidate["annual_holding_cost"],
            "total_inventory_cost": best_candidate["total_inventory_cost"],
            "total_annual_cost": best_candidate["total_annual_cost"],
        },
        "tier_evaluations": tier_evaluations,
        "inputs": {
            "demand": d,
            "annual_demand": d,
            "order_cost": s,
            "holding_rate": h_pct,
            "holding_cost_pct": h_pct,
            "holding_cost": h_fixed,
            "fixed_holding_cost": h_fixed,
            "discount_schedule": tier_list,
        },
    }

    _b = best_candidate
    _subs = []
    for _t in tier_evaluations:
        if _t.get("total_annual_cost") is not None:
            _subs.append(
                f"Tier {_t['tier']}: H = {_t['holding_cost_per_unit']}; Q = {_t['candidate_qty']} ({_t['selection_logic']}); "
                f"TC = {d:,.2f} x {_t['unit_price']} + ({d:,.2f} / {_t['candidate_qty']}) x {s:,.2f} + ({_t['candidate_qty']} / 2) x {_t['holding_cost_per_unit']} "
                f"= {_t['annual_purchase_cost']:,.2f} + {_t['annual_ordering_cost']:,.2f} + {_t['annual_holding_cost']:,.2f} = {_t['total_annual_cost']:,.2f}"
            )
        else:
            _subs.append(f"Tier {_t['tier']}: no feasible candidate ({_t['selection_logic']}); unconstrained EOQ = {_t['unconstrained_eoq']}")
    _subs.append(f"Minimum TC = {_b['total_annual_cost']:,.2f} at Tier {_b['tier']}, Q = {_b['candidate_qty']}, price = {_b['unit_price']}")
    _recomputed = d * _b["unit_price"] + (d / _b["candidate_qty"]) * s + (_b["candidate_qty"] / 2.0) * _b["holding_cost_per_unit"]
    return _attach_worksheet(
        _result,
        variable_definitions={
            "D": "Annual demand (units/year)",
            "S": "Ordering cost per order ($/order)",
            "P_i": "Unit purchase price in tier i ($/unit)",
            "H_i": "Holding cost per unit per year in tier i ($/unit/year)" + (" = h x P_i" if h_pct is not None else " (fixed, same for all tiers)"),
            "Q_i": "Candidate order quantity for tier i (units)",
            "TC_i": "Total annual cost for tier i ($/year)",
        },
        formulas=([ "H_i = h x P_i" ] if h_pct is not None else []) + [
            "EOQ_i = sqrt(2 x D x S / H_i)",
            "Candidate Q_i = EOQ_i if it lies in the tier range; the tier minimum if EOQ_i is below the tier; none if EOQ_i is above the tier",
            "TC_i = D x P_i + (D / Q_i) x S + (Q_i / 2) x H_i",
            "Optimal tier = tier with minimum TC_i",
        ],
        formulas_substituted=_subs,
        checks=[
            _check("Selected TC is the minimum across all feasible tiers", min(c["total_annual_cost"] for c in valid_candidates), _b["total_annual_cost"], tol=1e-9),
            _check("Selected Q lies inside its tier range", True, boundary_verified, kind="bool"),
            _check("Recomputed TC of selected tier matches reported TC (display-rounding tolerance)", _b["total_annual_cost"], _recomputed, tol=1e-4),
            _check("At least one feasible candidate evaluated", 1, len(valid_candidates), kind="ge"),
        ],
    )


# ============================================================================
# 3. SAFETY STOCK AND REORDER POINT (ROP)
# ============================================================================

def calculate_rop(
    avg_demand: float,
    lead_time: float,
    std_demand: float = 0.0,
    std_lead_time: float = 0.0,
    z_score: Optional[float] = None,
    service_level: Optional[float] = None,
    unit_cost: Optional[float] = None,
    annual_holding_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate Safety Stock and Reorder Point (ROP) across four variability scenarios.

    Scenarios Supported:
    1. Constant demand and constant lead time (Deterministic):
       sigma_LT = 0, SS = 0, ROP = d * L
    2. Variable demand and constant lead time (std_L = 0):
       sigma_LT = std_demand * sqrt(L), SS = z * sigma_LT, ROP = d * L + SS
    3. Constant demand and variable lead time (std_demand = 0):
       sigma_LT = avg_demand * std_lead_time, SS = z * sigma_LT, ROP = d * L + SS
    4. Variable demand and variable lead time:
       sigma_LT = sqrt(L * (std_demand^2) + (avg_demand^2) * (std_lead_time^2))
       SS = z * sigma_LT
       ROP = d * L + SS

    Args:
        avg_demand: Average demand rate per period (d >= 0).
        lead_time: Average lead time in periods (L > 0).
        std_demand: Standard deviation of demand per period (sigma_d >= 0).
        std_lead_time: Standard deviation of lead time in periods (sigma_L >= 0).
        z_score: Standard normal safety factor z.
        service_level: Target Cycle Service Level as a decimal fraction (0 < CSL < 1).
        unit_cost: Cost per unit (optional, for safety stock holding cost).
        annual_holding_rate: Annual holding rate fraction (optional).

    Returns:
        Structured dictionary containing final_answer, intermediate_calculations,
        assumptions, inputs, results, and verification audit.
    """
    # 1. Strict Input Validation (Never invent missing inputs)
    if avg_demand is None:
        raise ValueError("Missing required input: 'avg_demand'. Missing inputs are not invented.")
    if lead_time is None:
        raise ValueError("Missing required input: 'lead_time'. Missing inputs are not invented.")
    if avg_demand < 0:
        raise ValueError(f"Average demand cannot be negative; got {avg_demand}")
    if lead_time <= 0:
        raise ValueError(f"Lead time must be strictly positive; got {lead_time}")
    if std_demand < 0 or std_lead_time < 0:
        raise ValueError("Standard deviations cannot be negative.")

    has_variability = (std_demand > 0 or std_lead_time > 0)

    # 2. Determine z-score (strictly required if variability exists)
    if has_variability:
        if z_score is not None:
            z = float(z_score)
            csl = float(stats.norm.cdf(z)) if z > 0 else 0.5
        elif service_level is not None:
            if not (0.0 < service_level < 1.0):
                raise ValueError(f"Service level must be between 0 and 1 exclusive; got {service_level}")
            z = float(stats.norm.ppf(service_level))
            csl = float(service_level)
        else:
            raise ValueError(
                "Variability is present (std_demand > 0 or std_lead_time > 0). "
                "Must provide either 'service_level' (e.g. 0.95) or 'z_score'. Missing numerical inputs are not invented."
            )
    else:
        # Deterministic: no safety factor needed
        z = 0.0
        csl = 1.0

    # 3. Identify Scenario and Intermediate Variance Calculations
    expected_lead_time_demand = avg_demand * lead_time
    var_d_term = lead_time * (std_demand ** 2)
    var_l_term = (avg_demand ** 2) * (std_lead_time ** 2)
    total_variance = var_d_term + var_l_term
    sigma_lt = math.sqrt(total_variance)

    if std_demand == 0.0 and std_lead_time == 0.0:
        scenario = "Constant Demand and Constant Lead Time (Deterministic)"
    elif std_demand > 0.0 and std_lead_time == 0.0:
        scenario = "Variable Demand and Constant Lead Time"
    elif std_demand == 0.0 and std_lead_time > 0.0:
        scenario = "Constant Demand and Variable Lead Time"
    else:
        scenario = "Variable Demand and Variable Lead Time"

    safety_stock = z * sigma_lt
    reorder_point = expected_lead_time_demand + safety_stock

    ss_holding_cost = None
    if unit_cost is not None and annual_holding_rate is not None:
        ss_holding_cost = safety_stock * unit_cost * annual_holding_rate

    intermediate = {
        "scenario": scenario,
        "expected_lead_time_demand": round(expected_lead_time_demand, 4),
        "demand_variance_component": round(var_d_term, 4),
        "lead_time_variance_component": round(var_l_term, 4),
        "total_lead_time_variance": round(total_variance, 4),
        "sigma_lead_time": round(sigma_lt, 4),
        "z_score": round(z, 4),
        "cycle_service_level": round(csl, 4),
        "exact_safety_stock": round(safety_stock, 4),
        "exact_reorder_point": round(reorder_point, 4),
    }

    final_answer = {
        "reorder_point": round(reorder_point, 2),
        "reorder_point_integer": math.ceil(reorder_point),
        "safety_stock": round(safety_stock, 2),
        "safety_stock_integer": math.ceil(safety_stock) if safety_stock > 0 else 0,
        "expected_lead_time_demand": round(expected_lead_time_demand, 2),
        "variability_scenario": scenario,
    }

    assumptions = [
        f"Inventory policy scenario: {scenario}",
        "Continuous review inventory control policy (s, Q)",
        "Demand in non-overlapping time periods is independent and identically distributed",
        "Lead times are independent and identically distributed random variables",
        "Total lead time demand approximates a normal distribution",
        "All stockouts during the replenishment cycle are completely backordered",
    ]

    service_level_valid = (0.0 < csl <= 1.0)
    ss_non_negative = safety_stock >= -1e-6
    rop_valid = reorder_point >= (expected_lead_time_demand - 1e-6)

    verification = {
        "status": "PASSED" if (service_level_valid and ss_non_negative and rop_valid) else "WARNING",
        "service_level_valid": service_level_valid,
        "ss_non_negative": ss_non_negative,
        "rop_ge_lead_time_demand": rop_valid,
        "notes": (
            f"Verified under '{scenario}': Safety Stock = {safety_stock:.2f} (z={z:.3f}, sigma_LT={sigma_lt:.2f}), "
            f"yielding ROP = {reorder_point:.2f} units to maintain {csl * 100:.1f}% service level."
        ),
    }

    _result = {
        "model": "Safety Stock & Reorder Point (ROP)",
        "final_answer": final_answer,
        "intermediate_calculations": intermediate,
        "assumptions": assumptions,
        "verification": verification,
        "inputs": {
            "avg_demand": avg_demand,
            "lead_time": lead_time,
            "std_demand": std_demand,
            "std_lead_time": std_lead_time,
            "service_level": round(csl, 4),
            "z_score": round(z, 4),
            "unit_cost": unit_cost,
            "annual_holding_rate": annual_holding_rate,
        },
        "results": {
            "expected_lead_time_demand": round(expected_lead_time_demand, 2),
            "lead_time_demand_std": round(sigma_lt, 2),
            "safety_stock": round(safety_stock, 2),
            "safety_stock_integer": math.ceil(safety_stock) if safety_stock > 0 else 0,
            "reorder_point": round(reorder_point, 2),
            "reorder_point_integer": math.ceil(reorder_point),
            "annual_safety_stock_holding_cost": round(ss_holding_cost, 2) if ss_holding_cost is not None else None,
        },
    }

    _hasvar = std_demand > 0 or std_lead_time > 0
    _checks = [
        _check("Variance components sum to sigma_LT squared", var_d_term + var_l_term, sigma_lt ** 2, tol=1e-9),
        _check("ROP minus safety stock equals expected lead-time demand", expected_lead_time_demand, reorder_point - safety_stock, tol=1e-9),
        _check("Safety stock is non-negative", 0.0, safety_stock, kind="ge"),
    ]
    if _hasvar:
        _checks.append(_check("Phi(z) equals the target cycle service level", csl, float(stats.norm.cdf(z)), tol=1e-6))
    return _attach_worksheet(
        _result,
        variable_definitions={
            "d": "Average demand per period",
            "L": "Average lead time (periods)",
            "sigma_d": "Standard deviation of demand per period",
            "sigma_L": "Standard deviation of lead time (periods)",
            "CSL": "Target cycle service level",
            "z": "Standard normal safety factor = Phi^-1(CSL)",
            "sigma_LT": "Standard deviation of demand during lead time",
            "SS": "Safety stock (units)",
            "ROP": "Reorder point (units)",
        },
        formulas=[
            "Lead-time demand = d x L",
            "sigma_LT = sqrt(L x sigma_d^2 + d^2 x sigma_L^2)",
            "z = Phi^-1(CSL)",
            "SS = z x sigma_LT",
            "ROP = d x L + SS",
            f"Scenario: {scenario}",
        ],
        formulas_substituted=[
            f"Lead-time demand = {avg_demand} x {lead_time} = {expected_lead_time_demand:,.4f}",
            f"sigma_LT = sqrt({lead_time} x {std_demand}^2 + {avg_demand}^2 x {std_lead_time}^2) = sqrt({var_d_term:,.4f} + {var_l_term:,.4f}) = sqrt({total_variance:,.4f}) = {sigma_lt:,.4f}",
            (f"z = Phi^-1({csl:.4f}) = {z:.4f}" if _hasvar else "z = 0 (no demand or lead-time variability, so no safety stock is needed)"),
            f"SS = {z:.4f} x {sigma_lt:,.4f} = {safety_stock:,.4f}",
            f"ROP = {expected_lead_time_demand:,.4f} + {safety_stock:,.4f} = {reorder_point:,.4f}",
        ],
        checks=_checks,
    )

# Alias for backwards compatibility
calculate_safety_stock_rop = calculate_rop


# ============================================================================
# 4. FIXED-PERIOD REVIEW (PERIODIC REVIEW / BASE STOCK)
# ============================================================================

def calculate_periodic_review(
    review_period: float,
    lead_time: float,
    avg_demand: float,
    std_demand: float = 0.0,
    current_inventory: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate the target inventory level (Base Stock) and replenishment order quantity.

    Formulas:
        Protection period = review period + lead time
        Target inventory (M) = demand during protection period + safety stock
        Order quantity = max(0, target inventory - inventory position)
        where inventory position = current_inventory + on_order - backorders

    Args:
        review_period: Review interval length in periods (P > 0).
        lead_time: Replenishment delivery lead time in periods (L >= 0).
        avg_demand: Expected demand per period (d >= 0).
        std_demand: Standard deviation of demand per period (sigma_d >= 0).
        current_inventory: On-hand physical stock at review moment.
        on_order: Stock currently in pipeline / in transit.
        backorders: Unfilled outstanding customer orders.
        service_level: Target Cycle Service Level decimal (0 < CSL < 1).
        z_score: Standard normal safety factor z.

    Returns:
        Structured dictionary containing final_answer, intermediate_calculations,
        assumptions, inputs, results, and verification audit.
    """
    # 1. Strict Input Validation (Never invent missing inputs)
    if review_period is None or review_period <= 0:
        raise ValueError("Missing or invalid 'review_period': must be strictly positive (P > 0).")
    if lead_time is None or lead_time < 0:
        raise ValueError("Missing or invalid 'lead_time': cannot be negative (L >= 0).")
    if avg_demand is None or avg_demand < 0:
        raise ValueError("Missing or invalid 'avg_demand': cannot be negative (d >= 0).")
    if std_demand < 0:
        raise ValueError("Standard deviation of demand cannot be negative.")

    # 2. Determine z-score (strictly required if demand uncertainty exists)
    if std_demand > 0:
        if z_score is not None:
            z = float(z_score)
            csl = float(stats.norm.cdf(z)) if z > 0 else 0.5
        elif service_level is not None:
            if not (0.0 < service_level < 1.0):
                raise ValueError(f"Service level must be between 0 and 1 exclusive; got {service_level}")
            z = float(stats.norm.ppf(service_level))
            csl = float(service_level)
        else:
            raise ValueError(
                "Demand standard deviation > 0. Must provide either 'service_level' or 'z_score'. "
                "Missing numerical inputs are not invented."
            )
    else:
        z = 0.0
        csl = 1.0

    # 3. Intermediate Calculations
    protection_period = review_period + lead_time
    demand_protection = avg_demand * protection_period
    sigma_protection = std_demand * math.sqrt(protection_period)
    safety_stock = z * sigma_protection

    target_inventory = demand_protection + safety_stock
    inventory_position = current_inventory + on_order - backorders
    order_quantity = max(0.0, target_inventory - inventory_position)

    intermediate = {
        "protection_period": round(protection_period, 4),
        "expected_demand_during_protection": round(demand_protection, 4),
        "sigma_during_protection": round(sigma_protection, 4),
        "z_score": round(z, 4),
        "cycle_service_level": round(csl, 4),
        "safety_stock": round(safety_stock, 4),
        "exact_target_inventory": round(target_inventory, 4),
        "effective_inventory_position": round(inventory_position, 4),
        "unconstrained_order_quantity": round(target_inventory - inventory_position, 4),
    }

    final_answer = {
        "order_quantity": round(order_quantity, 2),
        "order_quantity_integer": math.ceil(order_quantity),
        "target_inventory": round(target_inventory, 2),
        "target_inventory_integer": math.ceil(target_inventory),
        "safety_stock": round(safety_stock, 2),
        "protection_period": round(protection_period, 1),
    }

    assumptions = [
        "Periodic review inventory policy (P, M) or (R, S) order-up-to control",
        "Inventory counts and ordering happen strictly at discrete review moments every P periods",
        "Protection interval spans review period plus replenishment lead time (P + L)",
        "Demand during the protection period follows a stationary normal distribution",
        "Unmet customer demand is completely backordered until delivery",
    ]

    prot_valid = protection_period > 0
    target_ge_demand = target_inventory >= (demand_protection - 1e-6)
    order_non_neg = order_quantity >= 0.0

    verification = {
        "status": "PASSED" if (prot_valid and target_ge_demand and order_non_neg) else "WARNING",
        "protection_period_valid": prot_valid,
        "target_ge_expected_demand": target_ge_demand,
        "order_quantity_non_negative": order_non_neg,
        "notes": (
            f"Verified: Protection period of {protection_period:.1f} periods requires target "
            f"base stock of {target_inventory:.2f} units. Given effective inventory position of "
            f"{inventory_position:.2f}, recommended replenishment order is {order_quantity:.2f} units."
        ),
    }

    _result = {
        "model": "Fixed-Period Review (Periodic Review / Base Stock)",
        "final_answer": final_answer,
        "intermediate_calculations": intermediate,
        "assumptions": assumptions,
        "verification": verification,
        "inputs": {
            "review_period": review_period,
            "lead_time": lead_time,
            "protection_interval": protection_period,
            "avg_demand": avg_demand,
            "std_demand": std_demand,
            "service_level": round(csl, 4),
            "z_score": round(z, 4),
            "current_inventory": current_inventory,
            "on_order": on_order,
            "backorders": backorders,
            "effective_inventory_position": round(inventory_position, 2),
        },
        "results": {
            "expected_protection_demand": round(demand_protection, 2),
            "protection_demand_std": round(sigma_protection, 2),
            "safety_stock": round(safety_stock, 2),
            "target_level": round(target_inventory, 2),
            "target_level_integer": math.ceil(target_inventory),
            "order_quantity": round(order_quantity, 2),
            "order_quantity_integer": math.ceil(order_quantity),
        },
    }

    _checks = [
        _check("Protection period equals review period + lead time", review_period + lead_time, protection_period, tol=1e-9),
        _check("Target inventory minus safety stock equals expected protection-period demand", demand_protection, target_inventory - safety_stock, tol=1e-9),
        _check("Inventory position = on-hand + on-order - backorders", current_inventory + on_order - backorders, inventory_position, tol=1e-9),
        _check("Order quantity equals max(0, target - inventory position)", max(0.0, target_inventory - inventory_position), order_quantity, tol=1e-9),
        _check("Order quantity is non-negative", 0.0, order_quantity, kind="ge"),
    ]
    if std_demand > 0:
        _checks.append(_check("Phi(z) equals the target service level", csl, float(stats.norm.cdf(z)), tol=1e-6))
    return _attach_worksheet(
        _result,
        variable_definitions={
            "P": "Review period (periods)",
            "L": "Lead time (periods)",
            "d": "Average demand per period",
            "sigma_d": "Standard deviation of demand per period",
            "z": "Standard normal safety factor = Phi^-1(service level)",
            "SS": "Safety stock (units)",
            "T": "Target (order-up-to) inventory level (units)",
            "IP": "Inventory position = on-hand + on-order - backorders",
            "Q": "Order quantity now (units)",
        },
        formulas=[
            "Protection period = P + L",
            "Demand during protection period = d x (P + L)",
            "SS = z x sigma_d x sqrt(P + L)",
            "T = d x (P + L) + SS",
            "IP = on-hand + on-order - backorders",
            "Q = max(0, T - IP)",
        ],
        formulas_substituted=[
            f"Protection period = {review_period} + {lead_time} = {protection_period:,.4f}",
            f"Demand during protection period = {avg_demand} x {protection_period:,.4f} = {demand_protection:,.4f}",
            f"SS = {z:.4f} x {std_demand} x sqrt({protection_period:,.4f}) = {z:.4f} x {sigma_protection:,.4f} = {safety_stock:,.4f}",
            f"T = {demand_protection:,.4f} + {safety_stock:,.4f} = {target_inventory:,.4f}",
            f"IP = {current_inventory} + {on_order} - {backorders} = {inventory_position:,.4f}",
            f"Q = max(0, {target_inventory:,.4f} - {inventory_position:,.4f}) = {order_quantity:,.4f}",
        ],
        checks=_checks,
    )

# Alias for backwards compatibility
calculate_fixed_period_review = calculate_periodic_review


# ============================================================================
# 5. NEWSVENDOR MODEL (SINGLE-PERIOD / PERISHABLE)
# ============================================================================

def calculate_newsvendor(
    selling_price: float,
    cost: float,
    salvage_value: float = 0.0,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    discrete_demand: Optional[List[float]] = None,
    discrete_probabilities: Optional[List[float]] = None,
    penalty_cost: float = 0.0,
) -> Dict[str, Any]:
    """Calculate underage cost, overage cost, critical ratio, and recommended order quantity.

    Economic Formulations:
        Underage Cost (Cu) = selling_price - cost + penalty_cost
        Overage Cost (Co) = cost - salvage_value
        Critical Ratio (CR) = Cu / (Cu + Co)
        For normal demand: Q* = demand_mean + z * demand_std where z = norm.ppf(CR)

    Args:
        selling_price: Unit selling price (P > C).
        cost: Unit acquisition / manufacturing cost (C > S).
        salvage_value: Unit clearance / scrap recovery value (S >= 0).
        demand_mean: Expected demand for normal distribution (mu >= 0).
        demand_std: Standard deviation of demand (sigma >= 0).
        discrete_demand: Optional discrete demand levels list.
        discrete_probabilities: Optional discrete probabilities list.
        penalty_cost: Goodwill penalty or backorder penalty per unsatisfied unit.

    Returns:
        Structured dictionary containing final_answer, intermediate_calculations,
        assumptions, inputs, results, and verification audit.
    """
    # 1. Strict Input Validation (Never invent missing inputs)
    if selling_price is None or cost is None:
        raise ValueError("Missing required inputs: 'selling_price' and 'cost'. Missing inputs are not invented.")
    if selling_price <= cost:
        raise ValueError(f"Selling price (${selling_price}) must strictly exceed cost (${cost}).")
    if cost < salvage_value:
        raise ValueError(f"Cost (${cost}) cannot be less than salvage value (${salvage_value}).")
    if salvage_value < 0 or penalty_cost < 0:
        raise ValueError("Salvage value and penalty cost cannot be negative.")

    # 2. Marginal Economic Costs
    c_u = (selling_price - cost) + penalty_cost
    c_o = cost - salvage_value
    denominator = c_u + c_o

    if denominator <= 0:
        raise ValueError("Denominator (Cu + Co) must be strictly positive.")

    critical_ratio = c_u / denominator

    # 3. Intermediate Calculations by Distribution Type
    intermediate: Dict[str, Any] = {
        "cost_of_underage_cu": round(c_u, 4),
        "cost_of_overage_co": round(c_o, 4),
        "critical_ratio": round(critical_ratio, 6),
    }

    # Case A: Normal Distribution
    if demand_mean is not None and demand_std is not None:
        if demand_mean < 0 or demand_std < 0:
            raise ValueError("Demand mean and standard deviation cannot be negative.")

        mu = float(demand_mean)
        sigma = float(demand_std)

        if sigma > 0:
            z = float(stats.norm.ppf(critical_ratio))
            q_star = mu + (z * sigma)
            phi_z = float(stats.norm.pdf(z))
            phi_cum = float(stats.norm.cdf(z))
            # Unit normal loss function
            unit_normal_loss = phi_z - (z * (1.0 - phi_cum))
            expected_lost_sales = sigma * unit_normal_loss
            expected_sales = mu - expected_lost_sales
            expected_leftovers = max(0.0, q_star - expected_sales)
        else:
            z = 0.0
            q_star = mu
            expected_lost_sales = 0.0
            expected_sales = mu
            expected_leftovers = 0.0

        q_star_final = max(0.0, q_star)
        expected_revenue = selling_price * expected_sales
        expected_salvage = salvage_value * expected_leftovers
        total_order_cost = cost * q_star_final
        expected_penalty = penalty_cost * expected_lost_sales
        expected_profit = expected_revenue + expected_salvage - total_order_cost - expected_penalty

        intermediate.update({
            "distribution_type": "Normal Distribution",
            "z_score": round(z, 4),
            "expected_sales": round(expected_sales, 2),
            "expected_leftovers": round(expected_leftovers, 2),
            "expected_lost_sales": round(expected_lost_sales, 2),
            "expected_profit": round(expected_profit, 2),
        })

        recommended_q = round(q_star_final, 2)
        recommended_q_int = round(q_star_final)

    # Case B: Discrete Distribution
    elif discrete_demand is not None and discrete_probabilities is not None:
        if len(discrete_demand) != len(discrete_probabilities):
            raise ValueError("Discrete demand and probabilities lists must have matching length.")
        prob_sum = sum(discrete_probabilities)
        if abs(prob_sum - 1.0) > 1e-3:
            raise ValueError(f"Discrete probabilities must sum to 1.0; got {prob_sum}")

        pairs = sorted(zip(discrete_demand, discrete_probabilities), key=lambda x: x[0])
        cum_prob = 0.0
        q_discrete = pairs[0][0]

        for d_val, p_val in pairs:
            cum_prob += p_val
            if cum_prob >= critical_ratio:
                q_discrete = d_val
                break

        e_sales = sum(min(d_val, q_discrete) * p_val for d_val, p_val in pairs)
        e_leftovers = sum(max(0.0, q_discrete - d_val) * p_val for d_val, p_val in pairs)
        e_lost = sum(max(0.0, d_val - q_discrete) * p_val for d_val, p_val in pairs)
        e_profit = (selling_price * e_sales) + (salvage_value * e_leftovers) - (cost * q_discrete) - (penalty_cost * e_lost)

        intermediate.update({
            "distribution_type": "Discrete Distribution",
            "expected_sales": round(e_sales, 2),
            "expected_leftovers": round(e_leftovers, 2),
            "expected_lost_sales": round(e_lost, 2),
            "expected_profit": round(e_profit, 2),
        })

        recommended_q = q_discrete
        recommended_q_int = int(round(q_discrete))
    else:
        raise ValueError(
            "Missing demand parameters: must provide either (demand_mean and demand_std) "
            "or (discrete_demand and discrete_probabilities). Missing numerical inputs are not invented."
        )

    # 4. Final Answer
    final_answer = {
        "recommended_order_quantity": recommended_q,
        "recommended_order_quantity_integer": recommended_q_int,
        "cost_of_underage_cu": round(c_u, 2),
        "cost_of_overage_co": round(c_o, 2),
        "critical_ratio": round(critical_ratio, 4),
        "target_in_stock_probability": round(critical_ratio, 4),
        "expected_profit": intermediate["expected_profit"],
    }

    assumptions = [
        "Single-period decision horizon (perishable, seasonal, or obsolete inventory)",
        "Zero intra-period replenishment (order placed prior to demand realization)",
        "Underage and overage costs are linear per unit",
        "Salvage recovery applies to all unsold inventory with zero salvage transaction cost",
        "Customer demand distribution is stationary and independent of stock level",
    ]

    cr_valid = (0.0 < critical_ratio < 1.0)
    margins_valid = (c_u > 0 and c_o > 0)
    q_non_negative = recommended_q >= 0

    verification = {
        "status": "PASSED" if (cr_valid and margins_valid and q_non_negative) else "WARNING",
        "critical_ratio_valid": cr_valid,
        "margins_positive": margins_valid,
        "q_non_negative": q_non_negative,
        "notes": (
            f"Verified: Underage cost Cu=${c_u:.2f}, Overage cost Co=${c_o:.2f}, yielding "
            f"Critical Ratio CR = {critical_ratio:.4f}. Optimal quantity Q* = {recommended_q}."
        ),
    }

    _result = {
        "model": "Newsvendor Model (Single-Period / Perishable)",
        "final_answer": final_answer,
        "intermediate_calculations": intermediate,
        "assumptions": assumptions,
        "verification": verification,
        "inputs": {
            "selling_price": selling_price,
            "cost": cost,
            "salvage_value": salvage_value,
            "penalty_cost": penalty_cost,
            "demand_mean": demand_mean,
            "demand_std": demand_std,
        },
        "results": {
            "cost_of_underage_cu": round(c_u, 2),
            "cost_of_overage_co": round(c_o, 2),
            "critical_ratio": round(critical_ratio, 4),
            "in_stock_probability": round(critical_ratio, 4),
            "optimal_order_quantity": recommended_q,
            "optimal_order_quantity_integer": recommended_q_int,
            "expected_profit": intermediate["expected_profit"],
            "expected_sales": intermediate["expected_sales"],
            "expected_leftover_units": intermediate["expected_leftovers"],
            "expected_lost_sales": intermediate["expected_lost_sales"],
            "z_score": intermediate.get("z_score"),
        },
    }

    _normal = demand_mean is not None and demand_std is not None
    _checks = [
        _check("Cu equals selling price - cost + penalty cost", selling_price - cost + penalty_cost, c_u, tol=1e-9),
        _check("Co equals cost - salvage value", cost - salvage_value, c_o, tol=1e-9),
        _check("Critical ratio lies strictly between 0 and 1", True, 0.0 < critical_ratio < 1.0, kind="bool"),
        _check("Critical ratio equals Cu / (Cu + Co)", c_u / (c_u + c_o), critical_ratio, tol=1e-9),
        _check("Order quantity is non-negative", 0.0, recommended_q, kind="ge"),
    ]
    if _normal:
        if demand_std > 0:
            _checks.append(_check("Phi(z) equals the critical ratio", critical_ratio, float(stats.norm.cdf(z)), tol=1e-6))
            _checks.append(_check("Q* equals mean + z x std (before flooring at zero)", mu + z * sigma, q_star, tol=1e-9))
            _qline = f"Q* = {mu} + {z:.4f} x {sigma} = {q_star:,.4f}"
        else:
            _qline = f"Q* = mean demand = {mu} (zero demand variability)"
        _dist = [
            "z = Phi^-1(CR)",
            "Q* = mu + z x sigma",
            "Expected profit = P x E[sales] + S x E[leftover] - C x Q* - penalty x E[lost sales]",
        ]
        _dsub = [
            f"z = Phi^-1({critical_ratio:.6f}) = {z:.4f}",
            _qline,
            f"Expected profit = {selling_price} x {expected_sales:,.4f} + {salvage_value} x {expected_leftovers:,.4f} - {cost} x {q_star_final:,.4f} - {penalty_cost} x {expected_lost_sales:,.4f} = {expected_profit:,.4f}",
        ]
    else:
        _checks.append(_check("Cumulative probability at Q* is at least the critical ratio", critical_ratio, cum_prob, kind="ge"))
        _dist = [
            "Q* = smallest demand level whose cumulative probability >= CR",
            "Expected profit = P x E[sales] + S x E[leftover] - C x Q* - penalty x E[lost sales]",
        ]
        _dsub = [
            f"Cumulative probability at Q* = {cum_prob:.4f} >= CR = {critical_ratio:.6f}, so Q* = {q_discrete}",
            f"Expected profit = {selling_price} x {e_sales:,.4f} + {salvage_value} x {e_leftovers:,.4f} - {cost} x {q_discrete} - {penalty_cost} x {e_lost:,.4f} = {e_profit:,.4f}",
        ]
    return _attach_worksheet(
        _result,
        variable_definitions={
            "P": "Selling price per unit ($)",
            "C": "Purchase / unit cost ($)",
            "S": "Salvage value per unsold unit ($)",
            "penalty": "Shortage (goodwill) penalty per unmet unit ($)",
            "Cu": "Underage cost: profit lost per unit short ($)",
            "Co": "Overage cost: loss per unsold unit ($)",
            "CR": "Critical ratio = Cu / (Cu + Co)",
            "mu": "Mean demand (units)" if _normal else "n/a (discrete demand)",
            "sigma": "Standard deviation of demand (units)" if _normal else "n/a (discrete demand)",
            "Q*": "Recommended order quantity (units)",
        },
        formulas=[
            "Cu = P - C + penalty",
            "Co = C - S",
            "CR = Cu / (Cu + Co)",
        ] + _dist,
        formulas_substituted=[
            f"Cu = {selling_price} - {cost} + {penalty_cost} = {c_u:,.4f}",
            f"Co = {cost} - {salvage_value} = {c_o:,.4f}",
            f"CR = {c_u:,.4f} / ({c_u:,.4f} + {c_o:,.4f}) = {critical_ratio:.6f}",
        ] + _dsub,
        checks=_checks,
    )


# ============================================================================
# 6. MULTICRITERIA INVENTORY CLASSIFICATION (SALES x HITS, A/B/C/D)
# ============================================================================

CLASSIFICATION_PARAMETERS: Dict[str, Any] = {
    "sales_weight": 0.65,
    "hits_weight": 0.35,
    "pareto_cutoffs": {"A": (0, 60), "B": (61, 80), "C": (81, 90), "D": (91, 100)},
    "class_scores": {"A": 4, "B": 3, "C": 2, "D": 1},
    "final_cutoffs": {"A": "score >= 3.25", "B": "2.50 - 3.24", "C": "1.75 - 2.49", "D": "score < 1.75"},
}

CLASSIFICATION_REQUIRED_COLUMNS = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]
_CLASS_SCORE = CLASSIFICATION_PARAMETERS["class_scores"]


def _pareto_class_from_pct(pct: int) -> str:
    if pct <= 60:
        return "A"
    if pct <= 80:
        return "B"
    if pct <= 90:
        return "C"
    return "D"


def _final_class_from_score(score: float) -> str:
    if score >= 3.25:
        return "A"
    if score >= 2.50:
        return "B"
    if score >= 1.75:
        return "C"
    return "D"


def _pareto_classify(df, value_col: str, id_col: str, prefix: str):
    """Independent Pareto analysis of one criterion; returns Series indexed like df."""
    import numpy as np

    order = df.sort_values([value_col, "_ItemKey"], ascending=[False, True], kind="mergesort").index
    values = df.loc[order, value_col].astype(float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"Total of '{value_col}' must be positive to compute cumulative percentages.")
    cum_pct = values.cumsum() / total * 100.0
    # Round half up to a whole percent (1e-9 guards against float noise such as 59.4999999)
    cum_int = np.floor(cum_pct + 0.5 + 1e-9).astype(int)
    classes = cum_int.map(_pareto_class_from_pct)
    return cum_int.reindex(df.index), classes.reindex(df.index)


def classify_inventory_abcd(df):
    """Multicriteria A/B/C/D inventory classification using Sales ($) and Hits.

    Step 1: Two independent Pareto analyses (Sales, Hits). Cumulative % (including the
            item itself) is rounded to a whole percent; A: 0-60, B: 61-80, C: 81-90,
            D: 91-100. Ties are broken by Item Number ascending. Classes score A=4..D=1.
    Step 2: Weighted Score = 0.65 x Sales Score + 0.35 x Hits Score (2 decimals).
    Step 3: Final Class: A >= 3.25, B 2.50-3.24, C 1.75-2.49, D < 1.75.

    Each Item Number is a separate item (never grouped by Item Name). The input
    dataframe is not modified. Returns a new dataframe sorted by Weighted Score
    descending, then Sales descending (then Item Number ascending).
    """
    import pandas as pd

    missing = [c for c in CLASSIFICATION_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    if df.empty:
        raise ValueError("The dataframe contains no data rows.")

    work = df[CLASSIFICATION_REQUIRED_COLUMNS].copy().reset_index(drop=True)
    work["Item Name"] = work["Item Name"].astype(str).str.strip()
    work["UoM"] = work["UoM"].astype(str).str.strip()
    work["Item Number"] = work["Item Number"].astype(str).str.strip()
    # Tie-break key: numeric ordering when all Item Numbers are numeric, else text ordering
    _num = pd.to_numeric(work["Item Number"], errors="coerce")
    work["_ItemKey"] = _num if _num.notna().all() else work["Item Number"]

    for col in ("Sales ($)", "Hits"):
        if work[col].dtype == object:
            work[col] = work[col].astype(str).str.replace(r"[\$,\s]", "", regex=True)
        work[col] = pd.to_numeric(work[col], errors="coerce")
        bad = work[work[col].isna()]
        if not bad.empty:
            raise ValueError(
                f"{len(bad)} row(s) have non-numeric or missing '{col}' "
                f"(data row numbers: {', '.join(str(i + 2) for i in bad.index[:10])})."
            )
        if (work[col] < 0).any():
            raise ValueError(f"'{col}' contains negative values.")
    if work["Item Number"].duplicated().any():
        dups = work.loc[work["Item Number"].duplicated(), "Item Number"].unique()[:10]
        raise ValueError("Duplicate Item Number(s) found: " + ", ".join(map(str, dups)))

    work["Sales Cum %"], work["Sales Class"] = _pareto_classify(work, "Sales ($)", "Item Number", "Sales")
    work["Hits Cum %"], work["Hits Class"] = _pareto_classify(work, "Hits", "Item Number", "Hits")
    work["Sales Score"] = work["Sales Class"].map(_CLASS_SCORE)
    work["Hits Score"] = work["Hits Class"].map(_CLASS_SCORE)

    w_s = CLASSIFICATION_PARAMETERS["sales_weight"]
    w_h = CLASSIFICATION_PARAMETERS["hits_weight"]
    work["Weighted Score"] = (w_s * work["Sales Score"] + w_h * work["Hits Score"]).round(2)
    work["Final Class"] = work["Weighted Score"].map(_final_class_from_score)

    cols = [
        "Item Number", "Item Name", "UoM", "Sales ($)", "Sales Cum %", "Sales Class", "Sales Score",
        "Hits", "Hits Cum %", "Hits Class", "Hits Score", "Weighted Score", "Final Class",
    ]
    ordered = work.sort_values(
        ["Weighted Score", "Sales ($)", "_ItemKey"], ascending=[False, False, True], kind="mergesort"
    )
    return ordered[cols].reset_index(drop=True)


def summarize_classification(classified):
    """Summary by Final Class (items, total sales, % items, % sales) with a Total row."""
    import pandas as pd

    n = len(classified)
    total_sales = float(classified["Sales ($)"].sum())
    rows = []
    for cls in ["A", "B", "C", "D"]:
        sub = classified[classified["Final Class"] == cls]
        sales = float(sub["Sales ($)"].sum())
        rows.append({
            "Final Class": cls,
            "Items": int(len(sub)),
            "Total Sales ($)": sales,
            "% of Items": len(sub) / n * 100.0 if n else 0.0,
            "% of Total Sales": sales / total_sales * 100.0 if total_sales else 0.0,
        })
    rows.append({
        "Final Class": "Total",
        "Items": n,
        "Total Sales ($)": total_sales,
        "% of Items": 100.0 if n else 0.0,
        "% of Total Sales": 100.0 if total_sales else 0.0,
    })
    return pd.DataFrame(rows)


def class_counts(classified, column: str):
    """Count of items in each class (A-D, zero-filled) for 'Sales Class' or 'Hits Class'."""
    counts = classified[column].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    return counts.rename_axis(column).rename("Items").reset_index()
