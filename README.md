# Supply Chain Analytics Inventory Decision Agent

An autonomous, mathematically rigorous inventory analytics decision agent built to evaluate natural-language supply chain questions and compute verified operational decisions across five foundational inventory models.

---

## 🏛️ System Architecture

The agent strictly follows the end-to-end decision pipeline:

```
Customer Question
       │
       ▼
 1. Identify Model
       │
       ▼
 2. Extract Inputs
       │
       ▼
 3. Select Python Tool
       │
       ▼
 4. Calculate (Pure Python)
       │
       ▼
 5. Verify (Mathematical & Domain Audit)
       │
       ▼
 6. Explain (Executive Prescriptions & Insights)
```

> **Zero Hallucinated Arithmetic:** All numerical, statistical, and optimization calculations are executed strictly in pure Python (`inventory_tools.py`) using `scipy`, `numpy`, and analytical closed-form solutions.

---

## 📦 Supported Decision Models

| Model | Application Scope | Mathematical Objective |
| :--- | :--- | :--- |
| **1. Economic Order Quantity (EOQ)** | Continuous review deterministic demand | Balances ordering costs and holding costs to find global minimum $Q^* = \sqrt{\frac{2DS}{H}}$. |
| **2. Quantity Discount** | All-units supplier price break schedules | Evaluates feasible unconstrained EOQs and price break points across tiers to minimize total acquisition and inventory costs. |
| **3. Safety Stock & Reorder Point (ROP)** | Demand and lead-time stochastic variability | Determines buffer stock $SS = z \cdot \sigma_{DL}$ and replenishment trigger $ROP = (d \cdot L) + SS$ for target Cycle Service Levels. |
| **4. Fixed-Period Review** | Periodic review $(P, M)$ / Base-stock system | Protects against stockouts across review interval + lead time $(P + L)$, determining base stock target and dynamic order quantities. |
| **5. Newsvendor Model** | Single-period seasonal / perishable items | Maximizes expected profit under demand uncertainty by calculating the critical fractile $CR = \frac{C_u}{C_u + C_o}$ and optimal $Q^*$. |

---

## 📂 Project Structure

```
Inventory-Agent/
├── app.py                      # Dual-mode UI (Streamlit Web Dashboard + Interactive CLI)
├── inventory_agent.py          # 7-stage architectural decision engine & NLP extractor
├── inventory_tools.py          # Pure Python mathematical calculation routines & audits
├── tests/
│   └── test_inventory_tools.py # Comprehensive pytest test suite (19 tests)
├── requirements.txt            # Python environment dependencies
├── conftest.py                 # Pytest path configuration
└── README.md                   # Documentation and usage guide
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive CLI Application
```bash
python app.py
```
Or execute a single natural language question directly:
```bash
python app.py --query "Annual demand is 12000 units, ordering cost is $50, holding cost is $2.50 per unit. What is our EOQ?"
```

### 3. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
*(Or run `python app.py --web`)*

### 4. Run the Test Suite
```bash
pytest -v
```
All 19 unit tests, edge-case tests, and end-to-end agent pipeline queries will execute and verify mathematical accuracy.

## Multicriteria Inventory Classification (A/B/C/D)

Upload a CSV with `Item Number, Item Name, UoM, Sales ($), Hits` in the sidebar uploader. The app runs two independent Pareto analyses (Sales and Hits; cumulative % rounded to a whole percent; A 0-60, B 61-80, C 81-90, D 91-100; A=4..D=1), computes Weighted Score = 0.65 x Sales Score + 0.35 x Hits Score, and assigns the Final Class (A >= 3.25, B 2.50-3.24, C 1.75-2.49, D < 1.75). Parameters are fixed. Implemented in `inventory_tools.classify_inventory_abcd`.
