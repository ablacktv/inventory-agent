"""Comprehensive Test Suite for Deterministic Inventory Tools & Decision Agent.

Tests strictly verify that each tool:
- Validates inputs
- Returns structured results
- Returns intermediate calculations
- Returns the final answer
- Identifies operational & economic assumptions
- Never invents missing numerical inputs

Covers all 5 required tools:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (all 4 cases: constant D/L, variable D/const L, const D/variable L, variable D/L)
4. calculate_periodic_review() (protection period, target inventory, order quantity)
5. calculate_newsvendor() (Cu, Co, Critical Ratio, Q*)
Plus:
- Synonym recognition tests (ordering cost, setup expense, cost per order, etc.)
- Parameter distinction tests for all 5 models
- Gemini API key resolution and fallback integration
"""

import math
import os
import pytest
from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_periodic_review,
    calculate_newsvendor,
    # Aliases
    calculate_safety_stock_rop,
    calculate_fixed_period_review,
)
from inventory_agent import SupplyChainInventoryAgent, get_gemini_api_key


# ============================================================================
# 1. calculate_eoq() UNIT TESTS
# ============================================================================

class TestCalculateEOQ:
    """Deterministic tests for calculate_eoq()."""

    def test_eoq_textbook_calculation(self):
        """EOQ: Q = sqrt(2DS/H) with D=10,000, S=$50, H=$2."""
        res = calculate_eoq(demand=10000, order_cost=50, holding_cost=2)

        # 1. Structured Results
        assert "final_answer" in res
        assert "intermediate_calculations" in res
        assert "assumptions" in res
        assert "verification" in res

        # 2. Final Answer & Intermediate Calculations
        fa = res["final_answer"]
        inter = res["intermediate_calculations"]

        # Q* = sqrt(2 * 10000 * 50 / 2) = sqrt(500,000) ~ 707.11
        assert math.isclose(fa["order_quantity"], 707.11, abs_tol=0.01)
        assert fa["order_quantity_integer"] == 707
        assert math.isclose(inter["annual_ordering_cost"], inter["annual_holding_cost"], abs_tol=0.05)
        assert math.isclose(fa["total_annual_policy_cost"], 1414.21, abs_tol=0.05)
        assert math.isclose(inter["orders_per_year"], 14.14, abs_tol=0.05)

        # 3. Assumptions Identified
        assert len(res["assumptions"]) >= 4
        assert any("Continuous review" in a for a in res["assumptions"])
        assert any("Deterministic" in a for a in res["assumptions"])

    def test_eoq_with_holding_rate_and_unit_cost(self):
        """EOQ where H = holding_rate * unit_cost: D=1200, S=25, C=100, h=0.20 => H=20."""
        res = calculate_eoq(demand=1200, order_cost=25, holding_rate=0.20, unit_cost=100)
        fa = res["final_answer"]
        inter = res["intermediate_calculations"]
        assert inter["holding_cost_per_unit"] == 20.0
        # Q = sqrt(2 * 1200 * 25 / 20) = sqrt(3000) ~ 54.77
        assert math.isclose(fa["order_quantity"], 54.77, abs_tol=0.01)

    def test_eoq_input_validation_and_no_invented_inputs(self):
        """Verify strict input validation and zero invented numerical inputs."""
        # Negative demand
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_eoq(demand=-1000, order_cost=50, holding_cost=2)

        # Zero order cost
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_eoq(demand=1000, order_cost=0, holding_cost=2)

        # Missing demand
        with pytest.raises(ValueError, match="Missing required input: 'demand'"):
            calculate_eoq(order_cost=50, holding_cost=2)

        # Missing order cost
        with pytest.raises(ValueError, match="Missing required input: 'order_cost'"):
            calculate_eoq(demand=1000, holding_cost=2)

        # Missing holding cost (must not invent default H)
        with pytest.raises(ValueError, match="Missing holding cost specification"):
            calculate_eoq(demand=1000, order_cost=50)


# ============================================================================
# 2. calculate_quantity_discount() UNIT TESTS
# ============================================================================

class TestCalculateQuantityDiscount:
    """Deterministic tests for calculate_quantity_discount()."""

    @pytest.fixture
    def sample_tiers(self):
        return [
            {"tier": 1, "min_qty": 0, "max_qty": 499, "price": 50.0},
            {"tier": 2, "min_qty": 500, "max_qty": 999, "price": 45.0},
            {"tier": 3, "min_qty": 1000, "max_qty": None, "price": 40.0},
        ]

    def test_quantity_discount_percentage_holding_cost(self, sample_tiers):
        """Test with holding_rate (% holding cost): D=1000, S=40, h=0.20."""
        res = calculate_quantity_discount(
            demand=1000,
            order_cost=40,
            tiers=sample_tiers,
            holding_rate=0.20,
        )
        assert "final_answer" in res
        assert "intermediate_calculations" in res
        assert "assumptions" in res

        fa = res["final_answer"]
        inter = res["intermediate_calculations"]

        # Tier 3 is optimal at price break minimum 1,000 units:
        # Purchase = 40,000, Ordering = (1000/1000)*40 = 40, Holding = (1000/2)*(0.2*40) = 4,000 => TC = 44,040
        assert fa["optimal_tier"] == 3
        assert fa["optimal_order_quantity"] == 1000.0
        assert fa["unit_price"] == 40.0
        assert fa["minimum_total_annual_cost"] == 44040.0

        # Check intermediate calculations trace all 3 tiers
        assert len(inter["tier_evaluations"]) == 3
        assert any("All-units quantity discount" in a for a in res["assumptions"])

    def test_quantity_discount_fixed_holding_cost(self, sample_tiers):
        """Test with fixed holding cost per unit ($H = $8 constant)."""
        res = calculate_quantity_discount(
            demand=1000,
            order_cost=40,
            tiers=sample_tiers,
            holding_cost=8.0,
        )
        fa = res["final_answer"]
        assert fa["optimal_tier"] == 3
        assert fa["optimal_order_quantity"] == 1000.0
        # Purchase = 40,000, Ordering = 40, Holding = (1000/2)*8 = 4000 => TC = 44,040
        assert fa["minimum_total_annual_cost"] == 44040.0

    def test_quantity_discount_input_validation(self, sample_tiers):
        """Verify input validation and refusal to invent missing inputs."""
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_quantity_discount(demand=0, order_cost=50, tiers=sample_tiers, holding_rate=0.2)

        with pytest.raises(ValueError, match="non-empty list"):
            calculate_quantity_discount(demand=1000, order_cost=50, tiers=[], holding_rate=0.2)

        with pytest.raises(ValueError, match="Missing holding cost specification"):
            calculate_quantity_discount(demand=1000, order_cost=50, tiers=sample_tiers)


# ============================================================================
# 3. calculate_rop() UNIT TESTS
# ============================================================================

class TestCalculateROP:
    """Deterministic tests for calculate_rop() covering all 4 variability cases."""

    def test_case1_constant_demand_constant_lead_time(self):
        """Case 1: Constant demand and constant lead time (Deterministic).
        sigma_LT = 0, SS = 0, ROP = d * L
        """
        res = calculate_rop(avg_demand=50, lead_time=6, std_demand=0.0, std_lead_time=0.0)
        fa = res["final_answer"]
        inter = res["intermediate_calculations"]

        assert fa["safety_stock"] == 0.0
        assert fa["reorder_point"] == 300.0  # 50 * 6
        assert inter["sigma_lead_time"] == 0.0
        assert "Constant Demand and Constant Lead Time (Deterministic)" in fa["variability_scenario"]
        assert len(res["assumptions"]) >= 4

    def test_case2_variable_demand_constant_lead_time(self):
        """Case 2: Variable demand and constant lead time.
        sigma_LT = sigma_d * sqrt(L), SS = z * sigma_LT, ROP = d * L + SS
        d=100, L=4, sigma_d=10, std_L=0, CSL=0.95 (z ~ 1.6449)
        """
        res = calculate_rop(
            avg_demand=100,
            lead_time=4,
            std_demand=10,
            std_lead_time=0,
            service_level=0.95,
        )
        fa = res["final_answer"]
        inter = res["intermediate_calculations"]

        # Expected lead time demand = 100 * 4 = 400
        # sigma_LT = 10 * sqrt(4) = 20
        # SS = 1.64485 * 20 = 32.90
        # ROP = 400 + 32.90 = 432.90
        assert inter["expected_lead_time_demand"] == 400.0
        assert inter["sigma_lead_time"] == 20.0
        assert math.isclose(fa["safety_stock"], 32.90, abs_tol=0.05)
        assert math.isclose(fa["reorder_point"], 432.90, abs_tol=0.05)
        assert fa["reorder_point_integer"] == 433
        assert "Variable Demand and Constant Lead Time" in fa["variability_scenario"]

    def test_case3_constant_demand_variable_lead_time(self):
        """Case 3: Constant demand and variable lead time.
        sigma_LT = d * sigma_L, SS = z * sigma_LT, ROP = d * L + SS
        d=50, L=5, sigma_d=0, std_L=1.5, z=1.65
        """
        res = calculate_rop(
            avg_demand=50,
            lead_time=5,
            std_demand=0.0,
            std_lead_time=1.5,
            z_score=1.65,
        )
        fa = res["final_answer"]
        inter = res["intermediate_calculations"]

        # sigma_LT = 50 * 1.5 = 75
        # SS = 1.65 * 75 = 123.75
        # ROP = 50 * 5 + 123.75 = 373.75
        assert inter["sigma_lead_time"] == 75.0
        assert math.isclose(fa["safety_stock"], 123.75, abs_tol=0.01)
        assert math.isclose(fa["reorder_point"], 373.75, abs_tol=0.01)
        assert "Constant Demand and Variable Lead Time" in fa["variability_scenario"]

    def test_case4_variable_demand_variable_lead_time(self):
        """Case 4: Variable demand and variable lead time.
        sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        SS = z * sigma_LT
        ROP = d * L + SS
        d=100, L=4, sigma_d=10, sigma_L=1, CSL=0.95 (z ~ 1.64485)
        """
        res = calculate_rop(
            avg_demand=100,
            lead_time=4,
            std_demand=10,
            std_lead_time=1,
            service_level=0.95,
        )
        fa = res["final_answer"]
        inter = res["intermediate_calculations"]

        # Variance = 4*(10^2) + (100^2)*(1^2) = 400 + 10000 = 10400
        # sigma_LT = sqrt(10400) ~ 101.9804
        # SS = 1.64485 * 101.9804 ~ 167.74
        # ROP = 400 + 167.74 = 567.74
        assert inter["demand_variance_component"] == 400.0
        assert inter["lead_time_variance_component"] == 10000.0
        assert inter["total_lead_time_variance"] == 10400.0
        assert math.isclose(inter["sigma_lead_time"], 101.98, abs_tol=0.02)
        assert math.isclose(fa["safety_stock"], 167.74, abs_tol=0.1)
        assert math.isclose(fa["reorder_point"], 567.74, abs_tol=0.1)
        assert "Variable Demand and Variable Lead Time" in fa["variability_scenario"]

    def test_rop_no_invented_service_level(self):
        """When uncertainty exists, service level or z must be specified (never invented)."""
        with pytest.raises(ValueError, match="Must provide either 'service_level'"):
            calculate_rop(avg_demand=100, lead_time=5, std_demand=15)


# ============================================================================
# 4. calculate_periodic_review() UNIT TESTS
# ============================================================================

class TestCalculatePeriodicReview:
    """Deterministic tests for calculate_periodic_review()."""

    def test_periodic_review_formulas(self):
        """Protection period = review period + lead time
        Target inventory = demand during protection period + safety stock
        Order quantity = max(0, target inventory - inventory position)
        P=14, L=4 (protection=18), d=25, sigma_d=6, CSL=0.98 (z ~ 2.0537)
        Current on-hand=80, on-order=40, backorders=5
        """
        res = calculate_periodic_review(
            review_period=14,
            lead_time=4,
            avg_demand=25,
            std_demand=6,
            service_level=0.98,
            current_inventory=80,
            on_order=40,
            backorders=5,
        )
        assert "final_answer" in res
        assert "intermediate_calculations" in res
        assert "assumptions" in res

        fa = res["final_answer"]
        inter = res["intermediate_calculations"]

        # Protection period = 14 + 4 = 18
        assert fa["protection_period"] == 18.0
        # Expected demand during protection = 25 * 18 = 450
        assert inter["expected_demand_during_protection"] == 450.0
        # sigma_protection = 6 * sqrt(18) ~ 25.4558
        assert math.isclose(inter["sigma_during_protection"], 25.46, abs_tol=0.02)
        # SS = 2.05375 * 25.4558 ~ 52.28
        assert math.isclose(fa["safety_stock"], 52.28, abs_tol=0.05)
        # Target inventory M = 450 + 52.28 = 502.28
        assert math.isclose(fa["target_inventory"], 502.28, abs_tol=0.05)
        assert fa["target_inventory_integer"] == 503

        # Effective inventory position = 80 + 40 - 5 = 115
        assert inter["effective_inventory_position"] == 115.0
        # Order quantity = max(0, 502.28 - 115) = 387.28
        assert math.isclose(fa["order_quantity"], 387.28, abs_tol=0.05)
        assert fa["order_quantity_integer"] == 388

    def test_periodic_review_order_quantity_zero_when_inventory_sufficient(self):
        """When inventory position exceeds target, order quantity is 0 (max(0, ...))."""
        res = calculate_periodic_review(
            review_period=7,
            lead_time=3,
            avg_demand=10,
            std_demand=0.0,
            current_inventory=200,
        )
        # Protection period = 10, Target = 10 * 10 = 100.
        # Position = 200 => Q = max(0, 100 - 200) = 0
        assert res["final_answer"]["order_quantity"] == 0.0

    def test_periodic_review_no_invented_service_level(self):
        """When std_demand > 0, service level cannot be invented."""
        with pytest.raises(ValueError, match="Demand standard deviation > 0. Must provide either 'service_level' or 'z_score'"):
            calculate_periodic_review(review_period=7, lead_time=3, avg_demand=10, std_demand=5)


# ============================================================================
# 5. calculate_newsvendor() UNIT TESTS
# ============================================================================

class TestCalculateNewsvendor:
    """Deterministic tests for calculate_newsvendor()."""

    def test_newsvendor_normal_distribution(self):
        """Calculate underage cost, overage cost, critical ratio, and Q*.
        Selling price = $35, Cost = $15, Salvage = $5.
        Demand ~ Normal(mu=600, sigma=120).
        """
        res = calculate_newsvendor(
            selling_price=35,
            cost=15,
            salvage_value=5,
            demand_mean=600,
            demand_std=120,
        )
        assert "final_answer" in res
        assert "intermediate_calculations" in res
        assert "assumptions" in res

        fa = res["final_answer"]
        inter = res["intermediate_calculations"]

        # Cu = 35 - 15 = 20
        # Co = 15 - 5 = 10
        assert fa["cost_of_underage_cu"] == 20.0
        assert fa["cost_of_overage_co"] == 10.0
        # CR = 20 / (20 + 10) = 2/3 ~ 0.6667
        assert math.isclose(fa["critical_ratio"], 0.6667, abs_tol=0.0005)
        # z = norm.ppf(2/3) ~ 0.430727
        # Q* = 600 + 0.430727 * 120 ~ 651.69
        assert math.isclose(fa["recommended_order_quantity"], 651.69, abs_tol=0.1)
        assert fa["recommended_order_quantity_integer"] == 652
        assert inter["expected_profit"] > 0
        assert len(res["assumptions"]) >= 4

    def test_newsvendor_with_penalty_cost(self):
        """Newsvendor with goodwill loss penalty: P=50, C=30, S=10, Penalty=5."""
        # Cu = (50 - 30) + 5 = 25; Co = 30 - 10 = 20 => CR = 25 / 45 ~ 0.5556
        res = calculate_newsvendor(
            selling_price=50,
            cost=30,
            salvage_value=10,
            penalty_cost=5,
            demand_mean=100,
            demand_std=20,
        )
        assert res["final_answer"]["cost_of_underage_cu"] == 25.0
        assert res["final_answer"]["cost_of_overage_co"] == 20.0
        assert math.isclose(res["final_answer"]["critical_ratio"], 25 / 45, abs_tol=0.0001)

    def test_newsvendor_input_validation(self):
        """Verify economic boundary checks and no invented inputs."""
        # Selling price <= cost
        with pytest.raises(ValueError, match="strictly exceed cost"):
            calculate_newsvendor(selling_price=20, cost=25, salvage_value=5, demand_mean=100, demand_std=10)

        # Cost < salvage
        with pytest.raises(ValueError, match="cannot be less than salvage value"):
            calculate_newsvendor(selling_price=50, cost=20, salvage_value=25, demand_mean=100, demand_std=10)

        # Missing demand parameters
        with pytest.raises(ValueError, match="Missing demand parameters"):
            calculate_newsvendor(selling_price=50, cost=30)


# ============================================================================
# 6. ORDERING COST SYNONYMS & TERMINOLOGY RECOGNITION
# ============================================================================

class TestSynonymRecognition:
    """Verify recognition of common inventory terminology and ordering cost synonyms."""

    @pytest.fixture
    def agent(self):
        return SupplyChainInventoryAgent()

    @pytest.mark.parametrize(
        "synonym",
        [
            "ordering cost",
            "order cost",
            "setup cost",
            "setup expense",
            "cost per order",
            "cost of placing an order",
            "ordering/setup cost",
        ],
    )
    def test_eoq_ordering_cost_synonyms(self, agent, synonym):
        """Test each requested ordering cost synonym in EOQ."""
        query = (
            f"Annual demand is 12,000 units. The {synonym} is $50, and holding cost is $2.50 per unit. "
            "What is our EOQ?"
        )
        inputs, missing = agent.extract_inputs(query, SupplyChainInventoryAgent.MODEL_EOQ)
        assert "order_cost" in inputs, f"Failed to extract order cost with synonym '{synonym}'"
        assert inputs["order_cost"] == 50.0
        assert len(missing) == 0, f"Unexpected missing inputs for '{synonym}': {missing}"

        # End-to-end execution
        res = agent.process_question(query)
        assert res.success is True
        assert res.calculation_results["final_answer"]["order_quantity_integer"] == 693


# ============================================================================
# 7. PARAMETER DISTINCTION TESTS FOR ALL 5 MODELS
# ============================================================================

class TestDistinguishParameters:
    """Verify the agent properly distinguishes each specific inventory parameter."""

    @pytest.fixture
    def agent(self):
        return SupplyChainInventoryAgent()

    def test_distinguish_safety_stock_and_rop(self, agent):
        """Distinguish average demand, demand std dev, average lead time, lead-time std dev, service level."""
        query = (
            "Our average demand is 80 units per day with a demand standard deviation of 12. "
            "Average lead time is 5 days with a lead-time standard deviation of 1.5 days. "
            "We maintain a 95% service level. What is the safety stock and reorder point?"
        )
        inputs, missing = agent.extract_inputs(query, SupplyChainInventoryAgent.MODEL_SAFETY_STOCK_ROP)
        assert inputs["avg_demand"] == 80.0
        assert inputs["std_demand"] == 12.0
        assert inputs["lead_time"] == 5.0
        assert inputs["std_lead_time"] == 1.5
        assert inputs["service_level"] == 0.95
        assert len(missing) == 0

        res = agent.process_question(query)
        assert res.success is True
        assert res.calculation_results["final_answer"]["variability_scenario"] == "Variable Demand and Variable Lead Time"

    def test_distinguish_quantity_discount(self, agent):
        """Distinguish quantity break, unit price, ordering cost, fixed vs percentage holding cost."""
        query = (
            "Annual demand is 5000 units. Cost per order is $60. "
            "Carrying rate is 20% of unit price. "
            "Supplier price breaks: 0-499: $30, 500-999: $27, 1000+: $25."
        )
        inputs, missing = agent.extract_inputs(query, SupplyChainInventoryAgent.MODEL_QUANTITY_DISCOUNT)
        assert inputs["annual_demand"] == 5000.0
        assert inputs["order_cost"] == 60.0
        assert inputs["holding_rate"] == 0.20
        assert len(inputs["discount_schedule"]) == 3
        assert len(missing) == 0

        res = agent.process_question(query)
        assert res.success is True
        assert res.calculation_results["final_answer"]["optimal_order_quantity"] >= 500

    def test_distinguish_fixed_period_review(self, agent):
        """Distinguish demand, review period, lead time, on-hand, on-order, backorders."""
        query = (
            "We have a review period of 14 days and lead time of 4 days. Average demand is 25 units/day "
            "with standard deviation of 6. Target service level is 98%. "
            "We currently have 80 units on hand, 40 on order, and 5 backorders."
        )
        inputs, missing = agent.extract_inputs(query, SupplyChainInventoryAgent.MODEL_FIXED_PERIOD_REVIEW)
        assert inputs["review_period"] == 14.0
        assert inputs["lead_time"] == 4.0
        assert inputs["avg_demand"] == 25.0
        assert inputs["std_demand"] == 6.0
        assert inputs["service_level"] == 0.98
        assert inputs["current_inventory"] == 80.0
        assert inputs["on_order"] == 40.0
        assert inputs["backorders"] == 5.0
        assert len(missing) == 0

        res = agent.process_question(query)
        assert res.success is True
        assert res.calculation_results["final_answer"]["order_quantity_integer"] == 388

    def test_distinguish_newsvendor(self, agent):
        """Distinguish selling price, purchase cost, salvage value, penalty cost, mean demand, demand std dev."""
        query = (
            "Selling price is $35, unit purchase cost is $15, salvage value is $5, and penalty cost is $2. "
            "Expected demand is 600 units with demand standard deviation of 120."
        )
        inputs, missing = agent.extract_inputs(query, SupplyChainInventoryAgent.MODEL_NEWSVENDOR)
        assert inputs["selling_price"] == 35.0
        assert inputs["cost"] == 15.0
        assert inputs["salvage_value"] == 5.0
        assert inputs["penalty_cost"] == 2.0
        assert inputs["demand_mean"] == 600.0
        assert inputs["demand_std"] == 120.0
        assert len(missing) == 0

        res = agent.process_question(query)
        assert res.success is True
        # Cu = 35 - 15 + 2 = 22, Co = 15 - 5 = 10 => CR = 22/32 ~ 0.6875
        assert math.isclose(res.calculation_results["final_answer"]["critical_ratio"], 22 / 32, abs_tol=0.001)

    def test_examine_entire_question_before_declaring_missing(self, agent):
        """Ensure no parameter is reported as missing when present under synonyms anywhere in question."""
        query = (
            "We need to find the optimal order batch for a component. "
            "Our annual usage is 10,000 units. Each setup expense costs $80. "
            "The item cost is $20 apiece and our holding rate is 15%."
        )
        inputs, missing = agent.extract_inputs(query, SupplyChainInventoryAgent.MODEL_EOQ)
        assert "annual_demand" in inputs
        assert "order_cost" in inputs
        assert "holding_cost" in inputs  # Synthesized from 15% * $20 = $3.00
        assert len(missing) == 0

        res = agent.process_question(query)
        assert res.success is True
        # EOQ = sqrt(2 * 10000 * 80 / 3) ~ 730.297
        assert res.calculation_results["final_answer"]["order_quantity_integer"] == 730


# ============================================================================
# 8. GEMINI API KEY & FALLBACK TESTS
# ============================================================================

class TestGeminiAPIFallback:
    """Verify Gemini API key resolution and deterministic fallback."""

    def test_get_gemini_api_key_from_env(self, monkeypatch):
        """Verify API key is read from GEMINI_API_KEY environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_key_12345")
        assert get_gemini_api_key() == "test_key_12345"

    def test_get_gemini_api_key_when_unset(self, monkeypatch):
        """Verify None is returned when GEMINI_API_KEY is not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert get_gemini_api_key() is None

    def test_agent_falls_back_to_regex_when_no_api_key(self, monkeypatch):
        """Verify that agent defaults gracefully to deterministic regex extractor."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        agent = SupplyChainInventoryAgent()
        res = agent.process_question(
            "Annual demand is 5,000 units. Ordering cost is $40. Holding cost is $2 per unit. What is EOQ?"
        )
        assert res.success is True
        assert res.extraction_method == "Deterministic NLP & Regex Extractor"
        assert res.calculation_results["final_answer"]["order_quantity_integer"] == 447


# ============================================================================
# 9. END-TO-END INVENTORY DECISION AGENT INTEGRATION TESTS
# ============================================================================

class TestInventoryAgentIntegration:
    """Verify all 5 preset scenarios through the full decision pipeline."""

    @pytest.fixture
    def agent(self):
        return SupplyChainInventoryAgent()

    def test_agent_all_preset_questions(self, agent):
        """Test all sample queries through the 7-stage agent pipeline."""
        from app import SAMPLE_QUESTIONS
        for label, q_text in SAMPLE_QUESTIONS.items():
            res = agent.process_question(q_text)
            assert res.success is True, f"Failed on {label}: {res.error_message}"
            assert res.calculation_results is not None
            assert res.verification["status"] == "PASSED"
            assert "final_answer" in res.calculation_results
            assert "intermediate_calculations" in res.calculation_results
            assert "assumptions" in res.calculation_results
            assert len(res.explanation) > 100


# ============================================================================
# 10. STUDENT-VERIFIABLE WORKSHEET (8 SECTIONS) FOR EVERY TOOL
# ============================================================================

_TIERS = [
    {"tier": 1, "min_qty": 0, "max_qty": 499, "price": 50.0},
    {"tier": 2, "min_qty": 500, "max_qty": 999, "price": 45.0},
    {"tier": 3, "min_qty": 1000, "max_qty": None, "price": 40.0},
]

_WORKSHEET_CASES = {
    "eoq": lambda: calculate_eoq(demand=10000, order_cost=50, holding_cost=2),
    "eoq_rate": lambda: calculate_eoq(demand=1200, order_cost=25, holding_rate=0.2, unit_cost=100),
    "qd_pct": lambda: calculate_quantity_discount(demand=1000, order_cost=40, tiers=_TIERS, holding_rate=0.2),
    "qd_fixed": lambda: calculate_quantity_discount(demand=1000, order_cost=40, tiers=_TIERS, holding_cost=8),
    "rop_const": lambda: calculate_rop(avg_demand=50, lead_time=4),
    "rop_vd": lambda: calculate_rop(avg_demand=80, lead_time=5, std_demand=12, service_level=0.95),
    "rop_vl": lambda: calculate_rop(avg_demand=50, lead_time=4, std_lead_time=1, service_level=0.95),
    "rop_vdvl": lambda: calculate_rop(avg_demand=50, lead_time=4, std_demand=8, std_lead_time=1, service_level=0.95),
    "periodic": lambda: calculate_periodic_review(
        review_period=14, lead_time=4, avg_demand=25, std_demand=6, service_level=0.98,
        current_inventory=80, on_order=40, backorders=5),
    "periodic_zero": lambda: calculate_periodic_review(
        review_period=7, lead_time=2, avg_demand=10, std_demand=2, service_level=0.9,
        current_inventory=500, on_order=0, backorders=0),
    "newsvendor": lambda: calculate_newsvendor(
        selling_price=35, cost=15, salvage_value=5, demand_mean=600, demand_std=120),
    "newsvendor_discrete": lambda: calculate_newsvendor(
        selling_price=10, cost=4, salvage_value=1,
        discrete_demand=[50, 100, 150], discrete_probabilities=[0.3, 0.4, 0.3]),
}


class TestVerifiableWorksheet:
    """Every tool must return all 8 student-verification sections, produced by Python."""

    @pytest.mark.parametrize("case", list(_WORKSHEET_CASES))
    def test_all_eight_sections_present(self, case):
        res = _WORKSHEET_CASES[case]()
        assert res["inputs_extracted"], "1. Inputs Extracted"
        assert res["variable_definitions"], "2. Variable Definitions"
        assert res["formula"], "3. Formula"
        assert res["formula_substituted"], "4. Formula with Substituted Values"
        assert res["intermediate_calculations"], "5. Intermediate Calculations"
        assert res["final_answer"], "6. Final Answer"
        assert res["verification_checks"], "7. Verification Checks"
        assert res["assumptions"], "8. Assumptions"
        assert len(res["worksheet_sections"]) == 8

    @pytest.mark.parametrize("case", list(_WORKSHEET_CASES))
    def test_verification_checks_all_pass(self, case):
        res = _WORKSHEET_CASES[case]()
        for c in res["verification_checks"]:
            assert {"check", "expected", "actual", "passed"} <= set(c)
            assert c["passed"] is True, f"{case}: failed check '{c['check']}'"
        assert res["verification"]["checks_passed"] is True
        assert res["verification"]["status"] == "PASSED"

    def test_eoq_substitution_uses_tool_values(self):
        res = calculate_eoq(demand=10000, order_cost=50, holding_cost=2)
        text = " ".join(res["formula_substituted"])
        assert "10,000" in text and "707.1068" in text

    def test_rop_substitution_shows_variable_formula(self):
        res = calculate_rop(avg_demand=50, lead_time=4, std_demand=8, std_lead_time=1, service_level=0.95)
        text = " ".join(res["formula"] + res["formula_substituted"])
        assert "sigma_LT = sqrt(L x sigma_d^2 + d^2 x sigma_L^2)" in text
        sigma_lt = res["intermediate_calculations"]["sigma_lead_time"]
        assert f"{sigma_lt:,.4f}" in text

    def test_newsvendor_substitution_contains_cu_co_cr(self):
        res = calculate_newsvendor(selling_price=35, cost=15, salvage_value=5, demand_mean=600, demand_std=120)
        text = " ".join(res["formula_substituted"])
        assert "Cu = 35 - 15 + 0.0 = 20.0000" in text
        assert "Co = 15 - 5 = 10.0000" in text
        assert "0.666667" in text

    def test_missing_inputs_still_not_invented(self):
        with pytest.raises(ValueError):
            calculate_eoq(demand=1000, order_cost=50)

    def test_agent_explanation_contains_all_eight_sections(self):
        agent = SupplyChainInventoryAgent()
        res = agent.process_question(
            "Annual demand is 5,000 units. Ordering cost is $40. Holding cost is $2 per unit. What is EOQ?"
        )
        assert res.success is True
        exp = res.explanation
        for header in [
            "1. Inputs Extracted", "2. Variable Definitions", "3. Formula",
            "4. Formula with Substituted Values", "5. Intermediate Calculations",
            "6. Final Answer", "7. Verification Checks", "8. Assumptions",
        ]:
            assert header in exp, header


# ============================================================================
# 11. CSV UPLOAD ANALYSIS (pandas)
# ============================================================================

class TestUploadedCsvAnalysis:
    def _df(self):
        import io
        import pandas as pd
        csv = "Product,Category,Units,Unit Cost\nA,X,10,2.5\nB,X,30,4.0\nC,Y,20,1.0\n"
        return pd.read_csv(io.StringIO(csv))

    def test_analysis_values(self):
        from app import analyze_inventory_dataframe
        a = analyze_inventory_dataframe(self._df())
        assert a["total_units"] == 60
        assert math.isclose(a["avg_unit_cost"], (2.5 + 4.0 + 1.0) / 3)
        assert a["top_product"] == "B"
        assert a["units_by_category"].to_dict() == {"X": 40, "Y": 20}
        assert list(a["dataframe"]["Inventory Value"]) == [25.0, 120.0, 20.0]
        assert math.isclose(a["total_inventory_value"], 165.0)

    def test_input_dataframe_not_mutated(self):
        from app import analyze_inventory_dataframe
        df = self._df()
        analyze_inventory_dataframe(df)
        assert "Inventory Value" not in df.columns

    def test_non_numeric_rejected(self):
        from app import analyze_inventory_dataframe
        df = self._df().astype({"Units": object})
        df.loc[0, "Units"] = "abc"
        with pytest.raises(ValueError, match="non-numeric"):
            analyze_inventory_dataframe(df)


# ============================================================================
# MULTICRITERIA INVENTORY CLASSIFICATION (Sales x Hits)
# ============================================================================

import pandas as pd
import inventory_tools as _it


def _frame(rows):
    return pd.DataFrame(rows, columns=["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"])


def test_classification_hand_checked_example():
    df = _frame([
        ["I1", " Widget ", " EA ", 50, 5],
        ["I2", "Widget", "EA", 30, 15],
        ["I3", "Gadget", "EA", 15, 30],
        ["I4", "Gizmo", "EA", 5, 50],
    ])
    out = _it.classify_inventory_abcd(df).set_index("Item Number")
    assert out["Sales Cum %"].to_dict() == {"I1": 50, "I2": 80, "I3": 95, "I4": 100}
    assert out["Sales Class"].to_dict() == {"I1": "A", "I2": "B", "I3": "D", "I4": "D"}
    assert out["Hits Cum %"].to_dict() == {"I4": 50, "I3": 80, "I2": 95, "I1": 100}
    assert out["Hits Class"].to_dict() == {"I4": "A", "I3": "B", "I2": "D", "I1": "D"}
    assert out["Weighted Score"].to_dict() == {"I1": 2.95, "I2": 2.30, "I3": 1.70, "I4": 2.05}
    assert out["Final Class"].to_dict() == {"I1": "B", "I2": "C", "I3": "D", "I4": "C"}
    # Same Item Name stays as two separate items
    assert len(out) == 4
    # Sorted by weighted score descending
    assert list(_it.classify_inventory_abcd(df)["Item Number"]) == ["I1", "I2", "I4", "I3"]


def test_classification_pareto_boundaries():
    # Cumulative exactly 60 -> A, 80 -> B, 90 -> C, 100 -> D
    df = _frame([["A1", "a", "EA", 60, 1], ["A2", "b", "EA", 20, 1], ["A3", "c", "EA", 10, 1], ["A4", "d", "EA", 10, 1]])
    out = _it.classify_inventory_abcd(df).set_index("Item Number")
    assert out["Sales Cum %"].to_dict() == {"A1": 60, "A2": 80, "A3": 90, "A4": 100}
    assert out["Sales Class"].to_dict() == {"A1": "A", "A2": "B", "A3": "C", "A4": "D"}
    # 60.5% rounds to 61 -> B ; 60.4% rounds to 60 -> A
    out = _it.classify_inventory_abcd(_frame([["X", "x", "EA", 605, 1], ["Y", "y", "EA", 395, 1]])).set_index("Item Number")
    assert out.loc["X", "Sales Cum %"] == 61 and out.loc["X", "Sales Class"] == "B"
    out = _it.classify_inventory_abcd(_frame([["X", "x", "EA", 604, 1], ["Y", "y", "EA", 396, 1]])).set_index("Item Number")
    assert out.loc["X", "Sales Cum %"] == 60 and out.loc["X", "Sales Class"] == "A"


def test_final_class_score_boundaries():
    f = _it._final_class_from_score
    assert [f(s) for s in (3.25, 3.24, 2.50, 2.49, 1.75, 1.74)] == ["A", "B", "B", "C", "C", "D"]


def test_classification_tie_break_by_item_number():
    # Equal sales: lower Item Number is ranked first and absorbs the cumulative % first
    df = _frame([["B2", "n", "EA", 50, 1], ["A1", "n", "EA", 50, 1]])
    out = _it.classify_inventory_abcd(df).set_index("Item Number")
    assert out.loc["A1", "Sales Cum %"] == 50 and out.loc["B2", "Sales Cum %"] == 100


def test_classification_does_not_modify_input_and_validates():
    df = _frame([["I1", " n ", " EA ", 10, 1], ["I2", "m", "EA", 5, 2]])
    before = df.copy(deep=True)
    _it.classify_inventory_abcd(df)
    pd.testing.assert_frame_equal(df, before)
    with pytest.raises(ValueError):
        _it.classify_inventory_abcd(df.drop(columns=["Hits"]))
    with pytest.raises(ValueError):
        _it.classify_inventory_abcd(_frame([["I1", "n", "EA", "abc", 1]]))
    with pytest.raises(ValueError):
        _it.classify_inventory_abcd(_frame([["I1", "n", "EA", 0, 1]]))


def test_classification_summary_and_counts():
    df = _frame([["I1", "a", "EA", 50, 5], ["I2", "b", "EA", 30, 15], ["I3", "c", "EA", 15, 30], ["I4", "d", "EA", 5, 50]])
    c = _it.classify_inventory_abcd(df)
    s = _it.summarize_classification(c).set_index("Final Class")
    assert s.loc["B", "Items"] == 1 and s.loc["C", "Items"] == 2 and s.loc["A", "Items"] == 0
    assert s.loc["C", "Total Sales ($)"] == 35 and s.loc["Total", "Total Sales ($)"] == 100
    assert s.loc["C", "% of Items"] == 50 and s.loc["Total", "% of Total Sales"] == 100
    assert _it.class_counts(c, "Sales Class")["Items"].tolist() == [1, 1, 0, 2]


def test_classification_tie_break_is_numeric_for_numeric_ids():
    df = _frame([[100, "n", "EA", 50, 1], [98, "n", "EA", 50, 1]])
    out = _it.classify_inventory_abcd(df).set_index("Item Number")
    assert out.loc["98", "Sales Cum %"] == 50 and out.loc["100", "Sales Cum %"] == 100


# ============================================================================
# TIME-UNIT NORMALIZATION (validation fix: monthly demand vs annual holding cost)
# ============================================================================

class TestTimeUnitNormalization:
    @pytest.fixture
    def agent(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        return SupplyChainInventoryAgent()

    def test_monthly_demand_annual_holding_eoq(self, agent):
        res = agent.process_question(
            "Find EOQ. Demand is 1,000 units per month, ordering cost is $25 per order, "
            "and holding cost is $0.50 per unit per year."
        )
        assert res.success is True
        fa = res.calculation_results["final_answer"]
        assert math.isclose(fa["order_quantity"], 1095.45, abs_tol=0.01)
        assert math.isclose(fa["total_annual_policy_cost"], 547.72, abs_tol=0.01)
        assert "Unit conversion" in res.explanation

    def test_weekly_and_daily_demand_converted_to_annual(self, agent):
        r = agent.process_question("Find EOQ. Demand is 200 units per week, ordering cost is $40, holding cost is $4 per unit per year.")
        assert math.isclose(r.calculation_results["final_answer"]["order_quantity"], math.sqrt(2 * 200 * 52 * 40 / 4), abs_tol=0.01)
        r = agent.process_question("Find EOQ. Demand is 20 units per day, ordering cost is $40, holding cost is $4 per unit per year.")
        assert math.isclose(r.calculation_results["final_answer"]["order_quantity"], math.sqrt(2 * 20 * 365 * 40 / 4), abs_tol=0.01)

    def test_annual_demand_not_converted(self, agent):
        r = agent.process_question("Annual demand is 10,000 units, setup expense is $50, and it costs $2 per unit per year to hold inventory. What is the EOQ?")
        assert math.isclose(r.calculation_results["final_answer"]["order_quantity"], 707.11, abs_tol=0.01)

    def test_already_converted_value_not_doubled(self, agent):
        inputs, notes = agent.normalize_time_units(
            "Demand is 1,000 units per month, holding cost $0.50 per unit per year.",
            SupplyChainInventoryAgent.MODEL_EOQ, {"demand": 12000.0, "order_cost": 25, "holding_cost": 0.5})
        assert inputs["demand"] == 12000.0

    def test_rop_lead_time_weeks_with_daily_demand(self, agent):
        q = ("Average demand is 40 units per day with a demand standard deviation of 5. "
             "Lead time is 2 weeks. Service level is 95%. What is the safety stock and reorder point?")
        r = agent.process_question(q)
        inter = r.calculation_results["intermediate_calculations"]
        assert math.isclose(inter["expected_lead_time_demand"], 40 * 14, abs_tol=0.01)
