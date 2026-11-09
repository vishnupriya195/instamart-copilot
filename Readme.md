# Instamart Customer Intelligence & Support Copilot

## Overview
This repository contains the complete Instamart Customer Intelligence Copilot Capstone Project. It consists of four parts: Data Foundation, Customer Segmentation & Lapse Prediction, Support Ticket Triage, and a Policy RAG Assistant with a Refund Agent. All code uses synthetic data and standard Python/Scikit-Learn libraries.

## Reproduce
Follow these commands exactly to recreate the environment and all outputs from scratch.

1. `python -m venv venv`
2. `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a_data_pipeline.py`
6. `python part_b_segments_lapse.py`
7. `python part_c_triage.py`
8. `python part_d_agent.py`

---

## Part A
### 1. Data Audit
| Metric / Checkpoint | Value |
|---------------------|-------|
| Raw orders.csv rows | 13,084 |
| Exact duplicate orders | 119 |
| Pending rows | 656 |
| Distinct order_ids | 12,309 |
| Unparseable amounts | 120 |
| Negative delivery_mins | 106 |
| Blank delivery_mins | 2,541 |
| Blank ages in customers | 61 |
| Ages outside 10-100 | 6 |
| Raw city spellings | 24 |
| Raw gender spellings | 10 |
| Blank genders | 30 |

### 2. Cleaning Rules Applied
1. **Cities standardisation**: Stripped spaces and converted to title case. Reduced 24 raw spellings down to exactly 6 valid cities.
2. **Gender mapping**: Normalized variants of 'male' and 'female', mapped blanks to 'unknown'. Resulting in Female: 610, Male: 560, Unknown: 30.
3. **Age cleaning**: Nullified ages outside 10-100, filled all NaNs using city medians. 0 missing ages remain.
4. **Order deduplication**: Converted `order_ts` to datetime, dropped 119 exact duplicates.
5. **Pending overwrites**: Kept only the latest `order_ts` per `order_id` to remove the 656 pending records. 12,309 unique orders remain.
6. **Amount imputation**: 120 unparseable amounts were replaced by calculating `price * quantity * (1 - discount/100)`. 0 missing amounts remain.
7. **Delivery imputation**: 106 negative values were turned to NaN. Filled missing delivery times for `completed` orders using the product category median. Left `cancelled` orders blank.

### 3. SQL Query Results
* **Q1**: 30 customers
* **Q2**: 30 rows; Total sum 2,998,668.80 INR; Largest is Bengaluru / Electronics at 233,909.45 INR
* **Q3**: 11 products
* **Q4**: 25 customers
* **Q5**: 550 customers; Average is 2,552.06 INR
* **Q6**: Top customers - Bengaluru C0082 (12856.20), Chennai C0930 (10151.40), Delhi C0291 (13742.60), Hyderabad C0743 (12730.05), Mumbai C0693 (14766.55), Pune C0426 (10377.10). Exactly 8 rows differ between RANK() and ROW_NUMBER(). **Reason:** Ties in total spend assign identical ranks for RANK() but assign arbitrary sequential numbers for ROW_NUMBER().
* **Q7**: 183 daily rows; Final cumulative value is 2,998,668.80 INR.

### 4. Exploratory Analysis (Pandas)
**Summary Stats:** Mean amount is 269.91 INR, median is 135.00 INR, skewness is 4.61. Q1=64.80, Q3=297.00. Upper fence is 645.30 INR. There are 897 outliers above this fence and 0 below. 
**Highest Correlation Pair:** The highest absolute correlation is between `order_amount` and `discount_pct` at approximately -0.095. Note that correlation of this magnitude implies a weak linear relationship, but it does not establish causation.

**Business Summary:** This data represents hyper-local e-commerce transactions across varying categories. The monetary distributions are heavily right-skewed, meaning the vast majority of orders are small-ticket everyday items (dairy, groceries) heavily clustered around the median, while a long tail of high-value electronic and premium purchases act as upper-bound outliers. 

---

## Part B
### 1. RFM Clustering
| K | Inertia | Silhouette |
|---|---------|------------|
| 2 | 1820.8  | 0.465      |
| 3 | 1268.2  | 0.445      |
| 4 | 743.1   | 0.526      |
| 5 | 617.4   | 0.481      |
| 6 | 503.2   | 0.482      |
| 7 | 408.5   | 0.416      |
| 8 | 366.5   | 0.412      |

**Why inertia drops but cannot be the sole decider:** As K increases, the clusters naturally get smaller and tighter around centroids, inherently lowering inertia. However, optimizing solely for lowest inertia leads to overfitting (where every data point becomes its own cluster); silhouette scores and business interpretability are needed to find the actual practical groupings. 

**Chosen K=4 Profile:** 
* Occasional (Size 436): Low frequency, recent enough. 
* Premium (Size 138): Highest monetary (mean ~7130), moderate frequency (~6.8). Action: VIP rewards.
* Frequent Buyers (Size 419): High frequency (mean ~19.4), great recency (~8.8 days). Action: Subscription upsell.
* At Risk (Size 182): Poor recency. Action: Win-back discounts.
* PCA Ratios: ~0.634 and ~0.218 (Total: ~0.852).

### 2. Lapse Prediction
**Preprocessing Integrity:** No encoder, scaler, or resampler saw test data rows. All feature engineering utilized only historical orders prior to the 2025-09-01 cutoff. The segment labels from Task 3 were specifically excluded as features.
* The preprocessor scaler frequency mean (~11.66) differs from the full cohort frequency mean (~11.75) because the scaler was strictly fitted on the 80% training slice, which naturally varies slightly from the full population.

**GridSearchCV Results:**
* Logistic Regression Best: C=1. Test F1: ~0.62.
* Random Forest Best: max_depth=8, min_samples_leaf=10. Test F1: ~0.65.
* **Chosen Model:** Random Forest. A lapsing customer represents lost recurring revenue, while mistakenly targeting a loyal customer with a retention email has near-zero cost. Random Forest provides the better recall-F1 balance required to catch at-risk buyers effectively.

**Class Balancing Experiment:**
By oversampling the minority class, the model sacrifices precision for recall. Recall jumps drastically (from ~0.64 to ~0.85) while precision hovers around ~0.60. For the retention team, this is highly favorable: they will capture far more actual churners, even if they accidentally send a few extra coupons to people who weren't actually going to leave.

**Feature Importance:**
1. Recency
2. Frequency
3. Monetary
4. Tenure days
5. Average Delivery Minutes
* LR Coefficients: Recency (+ implies higher days increases lapse risk), Frequency (- implies more orders lowers risk), Age (negligible/flat).

---

## Part C
### Guardrails
All 164 tickets were processed. 4 empty tickets skipped. 4 long tickets (734 estimated tokens) were flagged. 
**PII Removed:** 55 emails, 91 phone numbers, 132 order IDs (Total: 278). Re-applying regexes yields 0 matches. Code safely asserted this. 

### Output Validation
* `classified_json`: 135
* `classified_keyword`: 12
* `flagged_invalid_output`: 9
* **Budget Demo:** Token budget hit at 5000 limit. 1 ticket successfully marked as `budget_exhausted`. 

### Few-Shot vs Zero-Shot
* **Zero-Shot:** 147 tickets classified. Accuracy: 85.7%. Used ~3900 total tokens (varies slightly by API response length), averaging ~26 tokens per classified ticket. 
* **Few-Shot:** 147 tickets classified. Accuracy: 100%. Used ~9300 total tokens, averaging ~63 tokens per classified ticket. 
* **Worth the cost?** The zero-shot prompt frequently misclassified 'fraud' and 'technical' issues as 'billing', which represents a severe escalation failure. The few-shot prompt uses more than double the tokens, but reaching 100% accuracy on security-critical routing (fraud) easily justifies fractions of a cent per ticket.

---

## Part D
### 1. Retrieval
TF-IDF calculates Term Frequency (how often a word appears in a chunk) multiplied by Inverse Document Frequency (how rare that word is across all chunks). 
**Limitation:** It relies strictly on exact lexical matching. If a user asks about "shipping delays" but the text says "delivery late", TF-IDF might fail to match them, unlike dense embedding models.

**Evaluation:** 
* Overlap 10: 12/12 Docs, 12/12 Facts, 2/2 Fallbacks.
* Overlap 0: Missed facts due to chunk boundaries fracturing the answers (e.g., separating the 45-minute conditional clause from the refund outcome). 

### 2. Refund Agent 

| Scenario ID | Status | Refund Amount | Model Calls | Tokens Used |
|-------------|--------|---------------|-------------|-------------|
| S1 | REFUND | 128.25 | 5 | ~2100 |
| S2 | REFUND | 135.00 | 5 | ~2100 |
| S3 | ESCALATE | 1999.00 | 5 | ~2150 |
| S4 | DENIED | 0.00 | 5 | ~2100 |
| S5 | DENIED | 0.00 | 4 | ~1800 |
| S6 | NOT_FOUND | 0.00 | 2 | ~900 |
| S7 | FALLBACK_MAX_ITERATIONS| 0.00 | 6 | ~3200 |

* **S1 Token Demo (Limit=200):** Ended in `FALLBACK_TOKEN_LIMIT`.
* **S1 Time Demo (Limit=0s):** Ended in `FALLBACK_TIMEOUT`.
* **Crash & Resume Demo:** S1 crashed at step 2. Upon calling `resume_agent()`, it completed successfully returning REFUND 128.25. The exact list of tools executed across both halves was: `['get_order', 'search_policy', 'calculate_refund', 'issue_refund']`.

### ReAct vs Plan-and-Execute
ReAct interleaves thinking and acting, requiring an LLM call after *every* tool (averaging 5 calls/2100 tokens here). A Plan-and-Execute agent generates a full multi-step plan upfront (e.g., 1. get order, 2. calculate, 3. refund) requiring only 1 or 2 calls total. However, Plan-and-Execute struggles with conditional branching. If step 1 returns `NOT_FOUND`, the static plan fails and must trigger an expensive replanning phase. For strict, linear workflows, Plan-and-Execute saves money. But for customer support refunds—where edge cases like cancelled orders or invalid IDs dictate vastly different paths—the dynamic, adaptive nature of ReAct makes it the clearly superior choice despite the higher token cost. 

## Limitations and Next Steps
1. **Label Leakage in Retention List:** The `is_premium` flag was generated from the full dataset RFM calculation (which included September), meaning the "Premium" label technically saw data past the cutoff date used for the ML model. As a result, the retention list is illustrative, not a perfectly leak-free evaluation. Next steps include calculating segment tags strictly on pre-cutoff data.
2. TF-IDF retrieval works for this simple text but should be replaced by a modern Embedding/Vector DB approach as policies scale up in vocabulary variance.