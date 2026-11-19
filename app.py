"""Supply Chain Analytics Inventory Decision Agent - Application Entrypoint.

Architecture Pipeline:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Supports:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock & Reorder Point (ROP)
4. Fixed-Period Review (Periodic Review / Base Stock)
5. Newsvendor Model (Single-Period / Perishable)

Runs both as:
- Streamlit Web Application: `streamlit run app.py`
- Interactive CLI Application: `python app.py`
"""

import argparse
import json
import math
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from inventory_agent import SupplyChainInventoryAgent
import inventory_tools


# ============================================================================
# SAMPLE QUESTIONS FOR TESTING AND DEMONSTRATION
# ============================================================================

SAMPLE_QUESTIONS = {
    "1. Economic Order Quantity (EOQ)": (
        "Our warehouse experiences an annual demand of 15,000 units. "
        "Ordering cost is $60 per order, and annual holding cost is $3.00 per unit. "
        "Lead time is 6 days with 300 working days per year. What is our optimal EOQ and policy?"
    ),
    "2. Quantity Discount": (
        "We are evaluating a supplier quantity discount. Annual demand is 5,000 units and ordering cost is $50. "
        "Holding cost is 20% of unit price. The supplier offers these price breaks: "
        "0-499: $30.00, 500-999: $27.00, 1000+: $25.00. What is the optimal order quantity?"
    ),
    "3. Safety Stock and Reorder Point (ROP)": (
        "Average daily demand is 80 units with a standard deviation of 12 units. "
        "Replenishment lead time is 5 days. We target a 95% cycle service level. "
        "What safety stock buffer and reorder point should we maintain?"
    ),
    "4. Fixed-Period Review": (
        "We operate a periodic review system with reviews every 14 days and a supplier lead time of 4 days. "
        "Average daily demand is 25 units with a standard deviation of 6 units. We require a 98% service level. "
        "Currently we have 80 units on hand, 40 on order, and 5 backorders. What is our target stock and order quantity?"
    ),
    "5. Newsvendor (Perishable / Single-Period)": (
        "We are stocking commemorative apparel for an upcoming marathon. "
        "Selling price is $35 per unit, unit wholesale cost is $15, and unsold items can be liquidated for $5 salvage value. "
        "Demand is normally distributed with a mean of 600 units and a standard deviation of 120 units. "
        "How many units should we order to maximize expected profit?"
    ),
}


# ============================================================================
# UPLOADED CSV ANALYSIS (pandas)
# ============================================================================

REQUIRED_UPLOAD_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]


def analyze_inventory_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze an uploaded inventory dataframe using pandas only.

    Expects columns Product, Category, Units, Unit Cost. All results are computed
    from the dataframe passed in; nothing is hardcoded.
    """
    df = df.copy()
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
    df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce")
    bad = df[df["Units"].isna() | df["Unit Cost"].isna()]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} row(s) have non-numeric or missing values in 'Units' or 'Unit Cost' "
            f"(row numbers: {', '.join(str(i + 2) for i in bad.index[:10])}). Please fix the file and re-upload."
        )
    if df.empty:
        raise ValueError("The uploaded file contains no data rows.")

    df["Inventory Value"] = df["Units"] * df["Unit Cost"]
    top_idx = df["Units"].idxmax()
    return {
        "dataframe": df,
        "total_units": df["Units"].sum(),
        "avg_unit_cost": df["Unit Cost"].mean(),
        "top_product": df.loc[top_idx, "Product"],
        "top_product_units": df.loc[top_idx, "Units"],
        "units_by_category": df.groupby("Category")["Units"].sum(),
        "total_inventory_value": df["Inventory Value"].sum(),
    }


# ============================================================================
# MULTICRITERIA INVENTORY CLASSIFICATION (Streamlit rendering)
# ============================================================================

def render_classification_section(st, df_upload: pd.DataFrame) -> None:
    """Classify the already-uploaded dataframe (Sales x Hits) and display results."""
    st.subheader("🏷️ Multicriteria Inventory Classification (A/B/C/D)")
    try:
        classified = inventory_tools.classify_inventory_abcd(df_upload)
    except ValueError as e:
        st.error(f"Classification could not be performed: {e}")
        return

    params = inventory_tools.CLASSIFICATION_PARAMETERS
    with st.expander("Parameters used (fixed)", expanded=True):
        st.markdown(
            f"""
- **Weights:** Sales {params['sales_weight']:.2f}, Hits {params['hits_weight']:.2f}
- **Pareto cumulative % classes (Sales and Hits, analyzed independently):** A: 0-60, B: 61-80, C: 81-90, D: 91-100 (cumulative % includes the item itself, rounded to a whole percent; ties broken by Item Number ascending)
- **Class scores:** A=4, B=3, C=2, D=1
- **Weighted Score:** 0.65 × Sales Score + 0.35 × Hits Score (2 decimals)
- **Final Class:** A if score ≥ 3.25; B if 2.50-3.24; C if 1.75-2.49; D if below 1.75
"""
        )

    st.markdown("**Summary by Final Class**")
    summary = inventory_tools.summarize_classification(classified)
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%.2f"),
            "% of Items": st.column_config.NumberColumn("% of Items", format="%.2f%%"),
            "% of Total Sales": st.column_config.NumberColumn("% of Total Sales", format="%.2f%%"),
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Items in each Sales Class**")
        st.dataframe(inventory_tools.class_counts(classified, "Sales Class"), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Items in each Hits Class**")
        st.dataframe(inventory_tools.class_counts(classified, "Hits Class"), hide_index=True, use_container_width=True)

    st.markdown("**Classification Results** (sorted by Weighted Score, then Sales)")
    display_cols = [
        "Item Number", "Item Name", "Sales ($)", "Sales Cum %", "Sales Class", "Sales Score",
        "Hits", "Hits Cum %", "Hits Class", "Hits Score", "Weighted Score", "Final Class",
    ]
    st.dataframe(
        classified[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sales ($)": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
            "Sales Cum %": st.column_config.NumberColumn("Sales Cum %", format="%d%%"),
            "Hits Cum %": st.column_config.NumberColumn("Hits Cum %", format="%d%%"),
            "Weighted Score": st.column_config.NumberColumn("Weighted Score", format="%.2f"),
        },
    )
    st.download_button(
        "Download classification results (CSV)",
        classified[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="inventory_classification.csv",
        mime="text/csv",
    )


# ============================================================================
# STREAMLIT WEB APPLICATION
# ============================================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    agent = SupplyChainInventoryAgent()

    # Sidebar
    st.sidebar.title("📦 Inventory Agent")
    st.sidebar.markdown(
        """
        **Supply Chain Analytics Decision Engine**
        
        This agent interprets natural-language supply chain questions and executes analytical decision models using strict Python numerical routines:
        1. **EOQ** (Continuous Review)
        2. **Quantity Discount** (All-Units)
        3. **Safety Stock & ROP** (Variability)
        4. **Fixed-Period Review** (Periodic)
        5. **Newsvendor** (Single-Period)
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Decision Architecture:**\n"
        "1. 📥 Customer Question\n"
        "2. 🧠 Identify Model\n"
        "3. 🔍 Extract Inputs\n"
        "4. 🛠️ Select Python Tool\n"
        "5. 🧮 Calculate (Python)\n"
        "6. ✅ Verify (Audit)\n"
        "7. 💡 Explain & Prescribe"
    )

    # Data Upload (last section of the sidebar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Upload")
    uploaded_csv = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    # Main Header
    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        "*Autonomous inventory decision support system with mathematical verification and executive explanations.*"
    )

    tab_chat, tab_manual, tab_guide = st.tabs([
        "💬 Natural Language Decision Agent",
        "🧮 Direct Model Calculators",
        "📚 Inventory Model Guide & Formulas",
    ])

    # ------------------------------------------------------------------------
    # TAB 1: NATURAL LANGUAGE DECISION AGENT
    # ------------------------------------------------------------------------
    with tab_chat:
        st.subheader("Ask Any Inventory Management Question")
        st.caption("Type your operational question below, or select an executive scenario.")

        # Preset Buttons
        st.write("**Quick-Load Sample Scenarios:**")
        cols = st.columns(len(SAMPLE_QUESTIONS))
        selected_sample = None
        for i, (label, sample_text) in enumerate(SAMPLE_QUESTIONS.items()):
            short_name = label.split(". ")[1].split(" (")[0]
            if cols[i].button(short_name, key=f"btn_{i}"):
                selected_sample = sample_text

        # Text input
        default_val = selected_sample if selected_sample else ""
        if "user_query" not in st.session_state or selected_sample:
            st.session_state["user_query"] = default_val

        question = st.text_area(
            "Customer Inventory Inquiry:",
            value=st.session_state["user_query"],
            height=110,
            placeholder="e.g. Annual demand is 10000 units, order cost is $50, holding cost is $2. What is our EOQ?",
        )

        col_act1, col_act2 = st.columns([1, 5])
        run_btn = col_act1.button("🚀 Run Decision Agent", type="primary")

        if run_btn and question.strip():
            with st.spinner("Executing 7-stage inventory decision architecture..."):
                result = agent.process_question(question.strip())

            # Top Decision Card
            if result.success:
                st.success(f"**Identified Model:** {result.model_display_name} (Confidence: {result.model_confidence * 100:.1f}%)")
            else:
                st.warning(f"**Status:** {result.error_message}")

            # Architecture Pipeline Visualizer Expander
            with st.expander("🔬 View 7-Stage Architectural Pipeline Execution Trace", expanded=True):
                st.markdown("### Execution Pipeline Steps")
                step_cols = st.columns(len(result.execution_trace))
                for idx, step in enumerate(result.execution_trace):
                    status_icon = "✅" if step.status == "SUCCESS" or step.status == "PASSED" else ("⚠️" if step.status in ["PARTIAL", "WARNING"] else "❌")
                    with step_cols[idx]:
                        st.markdown(f"**Step {idx + 1}**\n\n{status_icon} **{step.step_name}**")
                        st.caption(step.description)

                st.markdown("---")
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.markdown("**📥 Extracted Parameters (Python Dict):**")
                    st.json(result.extracted_inputs)
                with pcol2:
                    st.markdown(f"**🛠️ Selected Python Analytical Engine:** `{result.selected_tool}`")
                    st.markdown("**✅ Verification Audit:**")
                    st.json(result.verification if result.verification else {"status": "Pending"})

            # Result Explanation
            st.markdown("---")
            st.markdown(result.explanation)

            # Contextual Visualization
            if result.success and result.calculation_results:
                st.markdown("---")
                st.subheader("📈 Decision Analytics Visualizer")
                m = result.identified_model

                if m == SupplyChainInventoryAgent.MODEL_EOQ:
                    # EOQ Cost Curve Chart
                    d = result.extracted_inputs["annual_demand"]
                    s = result.extracted_inputs["order_cost"]
                    h = result.calculation_results["inputs"]["holding_cost_per_unit"]
                    eoq = result.calculation_results["results"]["eoq"]

                    q_vals = np.linspace(max(10, eoq * 0.2), eoq * 2.2, 100)
                    ord_c = (d / q_vals) * s
                    hld_c = (q_vals / 2.0) * h
                    tot_c = ord_c + hld_c
                    df_chart = pd.DataFrame({
                        "Order Quantity (Q)": q_vals,
                        "Annual Ordering Cost": ord_c,
                        "Annual Holding Cost": hld_c,
                        "Total Policy Cost": tot_c,
                    }).set_index("Order Quantity (Q)")
                    st.line_chart(df_chart)
                    st.caption(f"Cost curve reaches strict global minimum at EOQ = {eoq:.1f} units.")

                elif m == SupplyChainInventoryAgent.MODEL_QUANTITY_DISCOUNT:
                    tiers = result.calculation_results["tier_evaluations"]
                    tier_rows = []
                    for t in tiers:
                        tier_rows.append({
                            "Tier": f"Tier {t['tier']}",
                            "Price ($)": t["unit_price"],
                            "Candidate Q": t["candidate_qty"],
                            "Purchase Cost ($)": t.get("annual_purchase_cost", 0),
                            "Holding Cost ($)": t.get("annual_holding_cost", 0),
                            "Ordering Cost ($)": t.get("annual_ordering_cost", 0),
                            "Total Cost ($)": t.get("total_annual_cost", 0),
                            "Selection Logic": t["selection_logic"],
                        })
                    df_tiers = pd.DataFrame(tier_rows)
                    st.dataframe(df_tiers, use_container_width=True)

                elif m == SupplyChainInventoryAgent.MODEL_SAFETY_STOCK_ROP:
                    r = result.calculation_results["results"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Expected Lead Time Demand", f"{r['expected_lead_time_demand']:,} units")
                    c2.metric("Safety Stock Buffer", f"{r['safety_stock_integer']:,} units", delta=f"{r['lead_time_demand_std']} std dev")
                    c3.metric("Reorder Point (ROP)", f"{r['reorder_point_integer']:,} units")

                elif m == SupplyChainInventoryAgent.MODEL_FIXED_PERIOD_REVIEW:
                    r = result.calculation_results["results"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Protection Interval", f"{result.extracted_inputs['protection_interval']} periods")
                    c2.metric("Target Level (M)", f"{r['target_level_integer']:,} units")
                    c3.metric("Order Quantity to Place", f"{r['order_quantity_integer']:,} units")

                elif m == SupplyChainInventoryAgent.MODEL_NEWSVENDOR:
                    r = result.calculation_results["results"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Critical Ratio", f"{r['critical_ratio'] * 100:.1f}%")
                    c2.metric("Optimal Order Q*", f"{r['optimal_order_quantity_integer']:,} units")
                    c3.metric("Expected Profit", f"${r['expected_profit']:,.2f}")
                    if "expected_value_of_perfect_info_evpi" in r:
                        c4.metric("EVPI", f"${r['expected_value_of_perfect_info_evpi']:,.2f}")

    # ------------------------------------------------------------------------
    # TAB 2: DIRECT MANUAL CALCULATORS
    # ------------------------------------------------------------------------
    with tab_manual:
        st.subheader("Direct Parameter Decision Calculators")
        calc_choice = st.selectbox(
            "Select Inventory Model Engine:",
            [
                "1. Economic Order Quantity (EOQ)",
                "2. Quantity Discount Analysis",
                "3. Safety Stock & Reorder Point (ROP)",
                "4. Fixed-Period Review (Base Stock)",
                "5. Newsvendor Model (Single-Period)",
            ],
        )

        if "1. Economic Order Quantity" in calc_choice:
            col1, col2 = st.columns(2)
            d = col1.number_input("Annual Demand (D)", min_value=1.0, value=10000.0, step=100.0)
            s = col2.number_input("Ordering / Setup Cost per Order ($S)", min_value=0.1, value=50.0, step=5.0)
            c = col1.number_input("Unit Purchase Price ($C, optional)", min_value=0.0, value=25.0, step=1.0)
            h = col2.number_input("Annual Holding Cost per Unit ($H)", min_value=0.01, value=2.5, step=0.1)
            lt = col1.number_input("Lead Time (days, optional)", min_value=0.0, value=5.0, step=1.0)

            if st.button("Calculate EOQ Policy"):
                res = inventory_tools.calculate_eoq(
                    annual_demand=d,
                    order_cost=s,
                    holding_cost_per_unit=h,
                    unit_price=c if c > 0 else None,
                    lead_time_days=lt if lt > 0 else None,
                )
                r = res["results"]
                st.write("---")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Optimal EOQ (Q*)", f"{r['eoq_integer']:,} units", f"{r['eoq']} exact")
                mc2.metric("Orders per Year", f"{r['orders_per_year']:.1f}")
                mc3.metric("Total Annual Policy Cost", f"${r['total_policy_cost']:,.2f}")
                if r.get("reorder_point"):
                    st.info(f"📍 **Reorder Point (ROP):** Place order when inventory reaches **{r['reorder_point']} units**.")
                st.json(res["verification"])

        elif "2. Quantity Discount" in calc_choice:
            st.markdown("Configure annual demand, ordering cost, and discount tiers:")
            qd_col1, qd_col2, qd_col3 = st.columns(3)
            d = qd_col1.number_input("Annual Demand (D)", min_value=1.0, value=5000.0)
            s = qd_col2.number_input("Order Cost ($S)", min_value=1.0, value=50.0)
            h_pct = qd_col3.number_input("Holding Rate (% of Price)", min_value=1.0, max_value=100.0, value=20.0) / 100.0

            st.write("**Price Schedule Tiers:**")
            t1_p = st.number_input("Tier 1 (0 - 499 units) Price ($)", value=30.0)
            t2_p = st.number_input("Tier 2 (500 - 999 units) Price ($)", value=27.0)
            t3_p = st.number_input("Tier 3 (1000+ units) Price ($)", value=25.0)

            if st.button("Evaluate Quantity Discounts"):
                schedule = [
                    {"tier": 1, "min_qty": 0, "max_qty": 499, "price": t1_p},
                    {"tier": 2, "min_qty": 500, "max_qty": 999, "price": t2_p},
                    {"tier": 3, "min_qty": 1000, "max_qty": None, "price": t3_p},
                ]
                res = inventory_tools.calculate_quantity_discount(
                    annual_demand=d,
                    order_cost=s,
                    discount_schedule=schedule,
                    holding_cost_pct=h_pct,
                )
                opt = res["optimal_decision"]
                st.success(f"⭐ **Optimal Recommendation:** Order **{opt['optimal_order_quantity']:,} units** in **Tier {opt['tier']}** @ ${opt['unit_price']:.2f}/unit (Total Cost: ${opt['total_annual_cost']:,.2f})")
                st.dataframe(pd.DataFrame(res["tier_evaluations"]))

        elif "3. Safety Stock" in calc_choice:
            sc1, sc2 = st.columns(2)
            d = sc1.number_input("Average Demand per Period (d)", min_value=0.0, value=80.0)
            lt = sc2.number_input("Lead Time (L)", min_value=0.1, value=5.0)
            sd = sc1.number_input("Demand Standard Deviation (σ_d)", min_value=0.0, value=12.0)
            slt = sc2.number_input("Lead Time Standard Deviation (σ_L)", min_value=0.0, value=0.0)
            csl = sc1.slider("Cycle Service Level (CSL)", min_value=50.0, max_value=99.9, value=95.0) / 100.0

            if st.button("Compute Safety Stock and ROP"):
                res = inventory_tools.calculate_rop(
                    avg_demand=d,
                    lead_time=lt,
                    std_demand=sd,
                    std_lead_time=slt,
                    service_level=csl,
                )
                r = res["results"]
                st.write("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Safety Stock", f"{r['safety_stock_integer']:,} units", f"{r['safety_stock']} exact")
                c2.metric("Reorder Point (ROP)", f"{r['reorder_point_integer']:,} units")
                c3.metric("Expected Lead Time Demand", f"{r['expected_lead_time_demand']:,} units")
                st.json(res["verification"])

        elif "4. Fixed-Period Review" in calc_choice:
            fc1, fc2 = st.columns(2)
            p = fc1.number_input("Review Period (P, in periods)", min_value=1.0, value=14.0)
            lt = fc2.number_input("Lead Time (L, in periods)", min_value=0.0, value=4.0)
            d = fc1.number_input("Average Demand Rate (d)", min_value=0.0, value=25.0)
            sd = fc2.number_input("Demand Standard Deviation (σ_d)", min_value=0.0, value=6.0)
            csl = fc1.slider("Target Service Level", min_value=50.0, max_value=99.9, value=98.0) / 100.0
            on_hand = fc2.number_input("Current On-Hand Inventory", min_value=0.0, value=80.0)
            on_ord = fc1.number_input("Inventory On Order", min_value=0.0, value=40.0)
            backord = fc2.number_input("Customer Backorders", min_value=0.0, value=5.0)

            if st.button("Calculate Base Stock Order"):
                res = inventory_tools.calculate_periodic_review(
                    review_period=p,
                    lead_time=lt,
                    avg_demand=d,
                    std_demand=sd,
                    service_level=csl,
                    current_inventory=on_hand,
                    on_order=on_ord,
                    backorders=backord,
                )
                r = res["results"]
                st.write("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Target Base Stock (M)", f"{r['target_level_integer']:,} units")
                c2.metric("Order Quantity to Place (Q)", f"{r['order_quantity_integer']:,} units")
                c3.metric("Protection Interval", f"{p + lt:.1f} periods")
                st.json(res["verification"])

        elif "5. Newsvendor" in calc_choice:
            nc1, nc2 = st.columns(2)
            sp = nc1.number_input("Selling Price ($P)", min_value=0.1, value=35.0)
            c = nc2.number_input("Unit Cost ($C)", min_value=0.1, value=15.0)
            sal = nc1.number_input("Salvage Value ($S)", min_value=0.0, value=5.0)
            mu = nc2.number_input("Mean Demand (μ)", min_value=0.0, value=600.0)
            sig = nc1.number_input("Demand Standard Deviation (σ)", min_value=0.0, value=120.0)

            if st.button("Optimize Newsvendor Stocking"):
                res = inventory_tools.calculate_newsvendor(
                    selling_price=sp,
                    cost=c,
                    salvage_value=sal,
                    demand_mean=mu,
                    demand_std=sig,
                )
                r = res["results"]
                st.write("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Critical Fractile (CR)", f"{r['critical_ratio'] * 100:.1f}%")
                c2.metric("Optimal Order (Q*)", f"{r['optimal_order_quantity_integer']:,} units")
                c3.metric("Expected Profit", f"${r['expected_profit']:,.2f}")
                c4.metric("EVPI", f"${r['expected_value_of_perfect_info_evpi']:,.2f}")
                st.json(res["verification"])

    # ------------------------------------------------------------------------
    # TAB 3: METHODOLOGY & MATHEMATICAL GUIDE
    # ------------------------------------------------------------------------
    with tab_guide:
        st.subheader("📚 Supply Chain Inventory Decision Model Formulations")
        st.markdown(
            """
            ### 1. Economic Order Quantity (EOQ)
            - **Objective:** Balances annual fixed order placement costs with physical inventory holding costs.
            - **Formula:** 
              $$Q^* = \\sqrt{\\frac{2 D S}{H}}$$
            - **Total Cost:** 
              $$TC(Q) = \\frac{D}{Q}S + \\frac{Q}{2}H$$
            - **Optimality Condition:** At $Q^*$, Annual Ordering Cost strictly equals Annual Holding Cost.

            ---

            ### 2. Quantity Discount Model
            - **Objective:** Evaluates whether price break discounts justify carrying higher average cycle stock.
            - **Algorithm:**
              1. Compute unconstrained EOQ for each tier price $P_i$.
              2. If EOQ falls inside tier boundary, evaluate total cost at $Q_i = EOQ_i$.
              3. If EOQ is below tier boundary, evaluate total cost at the minimum price break threshold.
              4. Pick $Q^*$ yielding absolute minimum Total Annual Acquisition & Inventory Cost.

            ---

            ### 3. Safety Stock & Reorder Point (ROP)
            - **Objective:** Protects against stockouts caused by demand and/or supplier lead time stochastic variability.
            - **Lead Time Variance:** 
              $$\\sigma_{DL}^2 = L \\sigma_d^2 + d^2 \\sigma_L^2$$
            - **Formulas:**
              $$SS = z \\times \\sigma_{DL}, \\quad ROP = (d \\times L) + SS$$
            - **Service Level:** $z = \\Phi^{-1}(\\text{CSL})$ where CSL is Cycle Service Level.

            ---

            ### 4. Fixed-Period Review (Base Stock Model)
            - **Objective:** Periodic inventory control (P-System) where stock is reviewed every $P$ intervals.
            - **Protection Interval:** Must protect across $P + L$ time units until the subsequent order arrives.
            - **Order-Up-To Level:**
              $$M = d(P + L) + z \\sigma_d \\sqrt{P + L}$$
            - **Replenishment Order:**
              $$Q = \\max(0, M - [\\text{On-Hand} + \\text{On-Order} - \\text{Backorders}])$$

            ---

            ### 5. Newsvendor Model (Single-Period / Perishable)
            - **Objective:** Maximizes expected profit for non-replenishable or seasonal goods with uncertain demand.
            - **Marginal Costs:**
              - Underage Cost: $C_u = P - C + \\text{Penalty}$ (lost contribution margin)
              - Overage Cost: $C_o = C - S$ (clearance loss)
            - **Critical Ratio:**
              $$CR = \\frac{C_u}{C_u + C_o}$$
            - **Optimal Quantity:** $Q^* = \\mu + z \\sigma$ where $z = \\Phi^{-1}(CR)$.
            """
        )

    # ------------------------------------------------------------------------
    # UPLOADED CSV DATA ANALYSIS (main page, below the existing tabs)
    # ------------------------------------------------------------------------
    if uploaded_csv is not None:
        st.markdown("---")
        st.header("📂 Uploaded Data Analysis")
        try:
            df_upload = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Could not read the uploaded file as CSV: {e}")
            df_upload = None

        if df_upload is not None:
            df_upload.columns = [str(c).strip() for c in df_upload.columns]
            missing_cols = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df_upload.columns]
            has_class_cols = all(c in df_upload.columns for c in inventory_tools.CLASSIFICATION_REQUIRED_COLUMNS)

            if has_class_cols:
                render_classification_section(st, df_upload)

            if missing_cols and has_class_cols:
                pass  # Classification-style file: the legacy Units/Unit Cost analysis does not apply.
            elif missing_cols:
                st.error(
                    "The uploaded file does not match a supported layout. Either include the "
                    "inventory-analysis columns: " + ", ".join(f"**{c}**" for c in REQUIRED_UPLOAD_COLUMNS)
                    + "; or the classification columns: "
                    + ", ".join(f"**{c}**" for c in inventory_tools.CLASSIFICATION_REQUIRED_COLUMNS)
                    + ". Missing for inventory analysis: " + ", ".join(missing_cols) + "."
                )
                st.subheader("Uploaded Data")
                st.dataframe(df_upload, use_container_width=True)
            else:
                try:
                    analysis = analyze_inventory_dataframe(df_upload)
                except ValueError as e:
                    st.error(str(e))
                    st.dataframe(df_upload, use_container_width=True)
                    analysis = None

                if analysis is not None:
                    st.subheader("Uploaded Data")
                    st.dataframe(df_upload, use_container_width=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Rows", f"{len(df_upload):,}")
                    c2.metric("Columns", f"{len(df_upload.columns):,}")
                    st.write("**Column names:** " + ", ".join(f"`{c}`" for c in df_upload.columns))

                    st.subheader("Results")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Units", f"{analysis['total_units']:,.0f}")
                    m2.metric("Average Unit Cost", f"${analysis['avg_unit_cost']:,.2f}")
                    m3.metric("Product with Highest Units", str(analysis["top_product"]),
                              f"{analysis['top_product_units']:,.0f} units", delta_color="off")
                    m4.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

                    st.subheader("Total Units by Category")
                    st.dataframe(analysis["units_by_category"].to_frame("Total Units"), use_container_width=True)

                    st.subheader("Data with Inventory Value")
                    st.dataframe(
                        analysis["dataframe"],
                        use_container_width=True,
                        column_config={
                            "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                            "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f"),
                        },
                    )


# ============================================================================
# CLI INTERACTIVE RUNNER
# ============================================================================

def run_cli_interactive():
    agent = SupplyChainInventoryAgent()

    print("\n" + "=" * 78)
    print("      SUPPLY CHAIN ANALYTICS INVENTORY DECISION AGENT")
    print("      Architecture: Question -> Identify -> Extract -> Tool -> Calculate -> Verify -> Explain")
    print("=" * 78)

    while True:
        print("\nSelect an option:")
        print("1. Run Sample 1: Economic Order Quantity (EOQ)")
        print("2. Run Sample 2: Quantity Discount Analysis")
        print("3. Run Sample 3: Safety Stock & Reorder Point (ROP)")
        print("4. Run Sample 4: Fixed-Period Review (Base Stock)")
        print("5. Run Sample 5: Newsvendor (Perishable / Single-Period)")
        print("6. Enter Custom Natural Language Inventory Question")
        print("7. Launch Streamlit Web Dashboard")
        print("8. Exit")

        choice = input("\nEnter choice (1-8): ").strip()

        if choice in ["1", "2", "3", "4", "5"]:
            keys = list(SAMPLE_QUESTIONS.keys())
            selected_key = keys[int(choice) - 1]
            q_text = SAMPLE_QUESTIONS[selected_key]
            print(f"\n[Running Scenario: {selected_key}]")
            print(f"Question: {q_text}\n")
            res = agent.process_question(q_text)
            _print_agent_result(res)

        elif choice == "6":
            print("\nEnter your inventory inquiry (or press enter to cancel):")
            user_q = input("> ").strip()
            if not user_q:
                continue
            res = agent.process_question(user_q)
            _print_agent_result(res)

        elif choice == "7":
            print("\nLaunching Streamlit web dashboard in background...")
            import subprocess
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])
            print("Streamlit dashboard initiated! Check your browser window.")
            break

        elif choice == "8":
            print("\nExiting Inventory Decision Agent. Goodbye!")
            break
        else:
            print("Invalid choice, please select 1 through 8.")


def _print_agent_result(res):
    print("-" * 78)
    print(f"IDENTIFIED MODEL: {res.model_display_name} (Confidence: {res.model_confidence * 100:.1f}%)")
    print(f"SELECTED TOOL:    inventory_tools.{res.selected_tool}")
    print(f"EXTRACTED INPUTS: {json.dumps(res.extracted_inputs, indent=2)}")
    print("-" * 78)
    print("EXECUTIVE DECISION & EXPLANATION:")
    print(res.explanation)
    print("-" * 78)
    if res.verification:
        print(f"VERIFICATION STATUS: {res.verification.get('status', 'N/A')}")
        print(f"VERIFICATION NOTES:  {res.verification.get('notes', 'N/A')}")
    print("=" * 78)


# ============================================================================
# MAIN DISPATCHER
# ============================================================================

if __name__ == "__main__":
    import streamlit

    if streamlit.runtime.exists():
        run_streamlit_app()
    else:
        parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
        parser.add_argument("--web", action="store_true", help="Launch Streamlit web dashboard")
        parser.add_argument("--query", type=str, help="Process a single question directly via CLI")
        args = parser.parse_args()

        if args.web:
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        elif args.query:
            agent = SupplyChainInventoryAgent()
            res = agent.process_question(args.query)
            _print_agent_result(res)
        else:
            run_cli_interactive()
