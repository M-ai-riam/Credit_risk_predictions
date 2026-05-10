Objective
The goal of this project was to build a predictive model to determine whether a loan application should be approved or denied (Loan_Status). By analyzing applicant data such as income, education, and credit history, we aim to automate the credit risk assessment process and identify the most critical factors in loan approval.

Approach
1. Data Cleaning & Preprocessing
Missing Value Imputation:

Categorical variables (Gender, Married, Self_Employed, Credit_History) were filled using the Mode.

Numerical variables (LoanAmount, Loan_Amount_Term) were filled using the Median to account for potential outliers.

Categorical Encoding:

Used LabelEncoder to convert text-based categories (Gender, Education, Married, etc.) into numerical values suitable for machine learning.

Feature Selection:

Selected key predictors for the model: ApplicantIncome, Education, Self_Employed, and Credit_History.

2. Modeling
Algorithm: Logistic Regression was used due to its efficiency in binary classification tasks (Approved vs. Denied).

Validation Strategy: The dataset was split into Training (80%) and Testing (20%) sets using train_test_split to ensure the model generalizes well to unseen data.

3. Evaluation Metrics
Accuracy Score: Used to measure the overall percentage of correct loan predictions.

Confusion Matrix: Generated to visualize the performance of the model and understand the balance between True Positives (correct approvals) and False Positives (incorrect approvals).

Results and Insights
Key Model Findings
Credit History is Critical: In this specific dataset, Credit_History often acts as the strongest predictor; applicants with a history of meeting credit obligations are significantly more likely to be approved.

Income vs. Status: Initial visualizations (Scatter Plots) suggest that while ApplicantIncome is a factor, it does not guarantee approval on its own without a solid credit history.

Model Performance: The Logistic Regression model provides a baseline accuracy that can be used to further refine risk thresholds for the lending institution.

Insights for the Business
Education Impact: The encoding of education allows the model to weigh whether specialized training correlates with lower default risks.

Streamlined Processing: This model can serve as a "first-pass" filter to automatically approve low-risk applicants, allowing loan officers to focus their time on more complex cases.
