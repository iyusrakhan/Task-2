📌 Task 1 – Bank Marketing Prediction (Classification)
Objective

Predict whether a customer will subscribe to a term deposit based on demographics, financial indicators, and campaign-related attributes.

Dataset

bank-additional-full.csv

Rows: 41,188

Target variable: y (yes/no for term deposit)

Positives: 4,640 (≈11.3%) → highly imbalanced dataset

Approach

EDA

Numeric cols: age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

Categorical cols: job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome

Class imbalance observed (majority = “no”).

Preprocessing

Standard scaling for numeric features.

One-hot encoding for categorical features.

Train/validation split with stratification.

Models

Logistic Regression (baseline linear model).

Random Forest (ensemble, non-linear).

Both with pipelines for clean preprocessing.

Evaluation

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.

Used stratified validation for fairness.

Results
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	~0.91	0.65	0.42	0.51	0.78
Random Forest	~0.93	0.73	0.49	0.59	0.82

⚡ Random Forest performed better overall, handling non-linearities and interactions between features.

Insights

duration (call length), euribor3m, and nr.employed were strong predictors.

Imbalance meant Recall was weaker — balancing techniques (SMOTE/undersampling) could further help.

📌 Task 2 – Customer Segmentation with Mall Data (Clustering)
Objective

Cluster mall customers based on demographic and spending behavior, then propose marketing strategies.

Dataset

Mall_Customers.csv

200 customers

Features: Gender, Age, Annual Income (k$), Spending Score (1-100)

Approach

EDA

Spending vs. Income scatterplot showed clear clusters.

Age and Gender distributions explored.

Preprocessing

Encoded Gender.

Standardized features before clustering.

Clustering

K-Means with optimal k=5 (Elbow & Silhouette).

PCA for 2D visualization.

Evaluation

Silhouette score ≈ 0.55 → good separation.

Results

5 clusters identified:

Low Income – Low Spending → Budget-conscious, not ideal targets.

High Income – Low Spending → Need premium engagement campaigns.

Medium Income – Medium Spending → Potentially nurture group.

Low Income – High Spending → Value-seekers, respond well to offers.

High Income – High Spending → VIP customers, focus on loyalty programs.

Insights

Clear customer segments found.

Marketing strategy can now be tailored:

VIPs → Exclusive offers, memberships.

Budget → Discounts, value packs.

Medium → Cross-selling & upselling campaigns.

📌 Task 3 – Energy Consumption Forecasting (Time Series)
Objective

Forecast short-term household power consumption using historical data.

Dataset

Household Power Consumption Dataset (Kaggle, .txt format parsed to hourly).

Resampled to hourly consumption.

Train: 34,254 hours (~3.9 years).

Test: 336 hours (14 days).

Approach

Preprocessing

Resampled to hourly frequency.

Created features: hour, dayofweek, is_weekend, lags (1h, 2h, 24h, 168h).

Models

ARIMA (univariate baseline).

Prophet (optional, not installed in env → skipped).

XGBoost (lag-based regression).

Evaluation

Metrics: MAE, RMSE.

Forecast plots compared to actual test values.

Results
Model	MAE	RMSE
ARIMA	0.675	0.829
Prophet	NaN	NaN
XGBoost	0.328	0.522

⚡ XGBoost clearly outperformed ARIMA, showing the benefit of engineered features.

Insights

ARIMA captured trend but missed complex seasonality.

Prophet was not run due to environment issues (installation too heavy).

XGBoost, with lag & calendar features, provided accurate short-term forecasts.

On average, prediction error was only ~0.33 kW, which is small relative to household usage (0–6 kW).
