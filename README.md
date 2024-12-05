# Loan-Approval-Prediction-By-Multiple-Models-ML-Project

Project Workflow
1. Loading and Exploring the Dataset
Imported required libraries like numpy, pandas, matplotlib, seaborn, and scikit-learn tools.
Loaded the dataset using pandas.read_csv().
Checked the dataset's structure using shape, columns, and info() methods.
Identified missing values and outliers for appropriate handling.

2. Data Cleaning and Preprocessing
Handling Missing Values:
Replaced missing values in numerical columns (LoanAmount, Loan_Amount_Term, and Credit_History) using median and mean.
Filled missing values in categorical columns (Gender, Married, Dependents, Self_Employed) using the mode of respective columns.
Outlier Detection:
Identified outliers using boxplots and handled them via normalization techniques.

3. Feature Engineering
Created new features:

TotalIncome: Sum of ApplicantIncome and CoapplicantIncome.
NewApplicantIncome: Log transformation of ApplicantIncome.
TotalLoanAmount: Sum of LoanAmount and Loan_Amount_Term.
Normalized LoanAmount and Loan_Amount_Term using log transformation.
Generated NewTotalIncome by applying log transformation to TotalIncome.
Dropped less significant columns:

ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Loan_ID, and TotalIncome.
Encoded categorical features using LabelEncoder.

4. Data Splitting
Split the dataset into dependent (y) and independent variables (x).
Used train_test_split() to divide the data into training (75%) and testing (25%) sets.

5. Model Training and Evaluation
Implemented and evaluated the following machine learning models:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
K-Nearest Neighbors Classifier
Evaluated models based on:
Accuracy Score
Classification Report (Precision, Recall, and F1-score)
Results
Accuracy Scores

Model	Accuracy (%)

Logistic Regression	: 81.17

Decision Tree Classifier : 73.38

Random Forest Classifier	: 79.87

K-Nearest Neighbors (KNN)	: 62.99

Classification Reports
Detailed classification reports are generated for each model, showing precision, recall, F1-score, and support.

Visualizations

Gender and Loan Applications: Bar plot of loan applications categorized by gender.

Marital Status and Loan Applications: Bar plot showing the relationship between marital status and loan applications.

Correlation Matrix: Heatmap to visualize correlations between numerical features.

Log Transformation Effects: Histograms and density plots demonstrating normalization effects on income and loan amount features.

Dependencies

Python 3.x

Libraries:

numpy

pandas

matplotlib

seaborn
scikit-learn
