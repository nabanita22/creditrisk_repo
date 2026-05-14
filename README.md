# Credit Risk Model

A machine learning project to predict whether a loan applicant will default or not, and assign them a credit rating (Poor / Average / Good / Excellent).

---

## What This Project Does

When a loan officer receives an application, they need to quickly decide:
- How likely is this person to default?
- What credit category do they fall in?

This model answers both questions automatically using the applicant's personal, loan, and bureau (credit history) data.

---

## Data Used

The project uses three datasets that are joined on `cust_id`:

**Customers** – Personal info of the borrower
- Age, Gender, Marital Status, Employment Status
- Income, Number of Dependents, Residence Type
- Years at current address, City, State, Zipcode

**Loans** – Details of the loan applied for
- Loan Purpose, Loan Type
- Sanction Amount, Loan Amount, Net Disbursement
- Loan Tenure, Principal Outstanding, Bank Balance
- Disbursal Date, Installment Start Date
- `default` ← **this is our target variable** (1 = defaulted, 0 = did not default)

**Bureau Data** – Credit history from the bureau (like CIBIL)
- No. of Open Accounts, No. of Closed Accounts
- Total Loan Months, Delinquent Months
- Total DPD (Days Past Due), Enquiry Count, Credit Utilization Ratio

---

## Project Flow

```
Raw Data (Customers + Loans + Bureau)
        ↓
   Data Merging (join on cust_id)
        ↓
   EDA (explore distributions, missing values, defaults)
        ↓
   Feature Engineering
   - delinquency_ratio = delinquent_months / total_loan_months
   - avg_dpd_per_delinquency = total_dpd / delinquent_months
   - loan_to_income = loan_amount / income
        ↓
   Data Preprocessing
   - Fix invalid values in loan_purpose (replace with mode)
   - Feature selection using IV (Information Value) + VIF
   - Min-Max Scaling for numeric features
        ↓
   Train / Test Split  →  75% Train | 25% Test
        ↓
   Model Training
   - Logistic Regression ✅ (selected)
   - XGBoost
   - Random Forest
        ↓
   Fine Tuning (RandomizedSearchCV + Optuna)
        ↓
   Model Evaluation (AUC, Gini, KS Statistic, Decile Table)
        ↓
   Credit Scorecard (Poor / Average / Good / Excellent)
        ↓
   Streamlit App (loan officer UI)
```

---

## Key Features Selected (by Information Value)

These are the variables that matter most in predicting default:

| Variable | IV | Why it matters |
|---|---|---|
| credit_utilization_ratio | 2.35 | High credit usage = high default risk |
| delinquency_ratio | 0.71 | More delinquencies = more risky |
| loan_to_income | 0.47 | High loan vs income = risky |
| avg_dpd_per_delinquency | 0.40 | More days overdue = risky |
| loan_purpose | 0.36 | Some purposes have higher default rates |
| residence_type | 0.24 | Moderate impact |
| loan_tenure_months | 0.21 | Longer loans = slightly riskier |

---

## Model Performance

All three models beat the success criteria (AUC > 85, Gini > 85, KS > 40).

**Logistic Regression was chosen** — it performs nearly as well as XGBoost but is far easier to explain to the business team, which was a hard requirement in the SOW.

| Model | AUC | Gini |
|---|---|---|
| **Logistic Regression** ✅ | **98%** | **96%** |
| XGBoost | 99% | 96% |
| Random Forest | 97% | 95% |

**Decile Analysis Result:**
- Max KS stat = **85.99** (found in decile 1, i.e., top 2nd bucket)
- Top 3 Decile Capture Rate = **99.53%** (model catches 99.5% of all defaulters in the top 3 deciles)
- This far exceeds the success criteria of KS > 40 in first 3 deciles

---

## Success Criteria – All Met ✅

| Criteria | Target | Achieved |
|---|---|---|
| AUC | > 85 | 98% |
| Gini | > 85 | 96% |
| KS Statistic | > 40 | 85.99 |
| Max KS in first 3 deciles | Yes | Yes (decile 1) |
| Model interpretability | High | Yes (Logistic Regression) |

---

## Credit Scorecard

The model's predicted default probability is converted into a score and bucketed into 4 categories:

| Rating | Meaning |
|---|---|
| Excellent | Very low default risk – approve |
| Good | Low default risk – likely approve |
| Average | Moderate risk – manual review |
| Poor | High default risk – likely reject |

---

## Streamlit App

A simple UI for loan officers where they:
1. Enter borrower details (age, income, employment, etc.)
2. Enter loan details (amount, tenure, purpose, etc.)
3. Enter bureau data (credit utilization, DPD, open accounts, etc.)
4. Click **Predict** → get default probability + credit rating instantly

---

## Repository Structure

```
credit-risk-model/
│
├── data/                  # Raw datasets (customers, loans, bureau)
├── notebooks/             # EDA and experimentation (eda branch)
├── src/
│   ├── preprocessing.py   # Data cleaning and feature engineering
│   ├── train.py           # Model training
│   ├── scorecard.py       # Score to rating mapping
│   └── predict.py         # Prediction pipeline
├── models/                # Saved model artifacts
├── app.py                 # Streamlit application
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone <repo-url>
cd credit-risk-model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python src/train.py

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## Tech Stack

- **Python** – core language
- **Pandas, NumPy** – data processing
- **Scikit-learn** – Logistic Regression, preprocessing, evaluation
- **XGBoost** – experimented but not selected
- **Optuna / RandomizedSearchCV** – hyperparameter tuning
- **Streamlit** – UI for loan officers
- **Matplotlib / Seaborn** – EDA visualizations
