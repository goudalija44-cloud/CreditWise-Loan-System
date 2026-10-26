# CreditWise Loan Approval System

A machine learning project that predicts whether a loan application will be **approved or rejected** based on the applicant's financial and personal details.

## Problem Statement
Banks receive many loan applications and need a fast, consistent way to assess them. This project builds and compares classification models to predict `Loan_Approved` (Yes/No).

## Dataset
- File: `loan_approval_data.csv`
- 1,000 records, 20 columns
- Features include: Applicant & Coapplicant Income, Credit Score, DTI Ratio, Savings, Collateral Value, Loan Amount, Loan Term, Employment Status, Education, Property Area, and more
- Target: `Loan_Approved`

## Workflow
1. **Data Cleaning**: handled missing values (mean for numerical, most frequent for categorical columns)
2. **EDA**: class balance, categorical counts, income distributions, boxplots, credit score vs. approval
3. **Feature Encoding**: Label Encoding and One-Hot Encoding
4. **Correlation Analysis**: heatmap of feature relationships (Credit Score and DTI Ratio correlate most with approval)
5. **Train-Test Split** (80/20) and **Standard Scaling**
6. **Model Training & Evaluation** using Precision, Recall, F1 and Accuracy
7. **Feature Engineering**: added `DTI_Ratio²` and `Credit_Score²` to improve performance

## Models Used
- Logistic Regression
- K-Nearest Neighbors (K = 7)
- Gaussian Naive Bayes

## Results (after feature engineering)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.88 | 0.78 | 0.84 | 0.81 |
| KNN | 0.77 | 0.65 | 0.51 | 0.57 |
| Naive Bayes | 0.86 | 0.81 | 0.70 | 0.75 |

- **Logistic Regression** gives the best balance of precision and recall.
- **Naive Bayes** gives the highest precision.

## Tech Stack
Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

## How to Run
```bash
# 1. Clone the repository
git clone https://github.com/goudalija44-cloud/CreditWise-Loan-System.git
cd CreditWise-Loan-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Loan_Project.ipynb
```
Make sure `loan_approval_data.csv` is in the same folder as the notebook.

## Project Structure
```
├── Loan_Project.ipynb
├── loan_approval_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Author
**<Shubhashree Gouda>**
GitHub: [@goudalija44-cloud](https://github.com/goudalija44-cloud)