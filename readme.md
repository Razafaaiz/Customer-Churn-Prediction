🛍️ Customer Churn Prediction (Bank Dataset)
📌 Project Overview

This project predicts customer churn (whether a customer will leave the bank) using Deep Learning (Keras + TensorFlow) and provides an interactive dashboard built in Streamlit.

It demonstrates end-to-end data science workflow:

Data preprocessing (encoding, scaling, cleaning).

Model training & evaluation.

Deployment as a web app with two tabs:

Prediction Tab → Predict churn probability for a single customer.

Insights Tab → Interactive charts to explore churn patterns across demographics.

🎯 Objective

Help banks identify customers most likely to leave.

Provide business insights to reduce churn.

Showcase skills in ML, data preprocessing, visualization, and deployment.

⚙️ Tech Stack

Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Pickle)

Streamlit (for UI & deployment)

Matplotlib/Seaborn/Plotly (visualization)

📊 Dataset

Source: Kaggle - Bank Customer Churn Dataset.

Features:

Demographics → Age, Gender, Geography

Banking details → CreditScore, Balance, Products, Tenure, Salary

Behavior → ActiveMember, HasCrCard

Target: Exited (1 = churn, 0 = stay)

🔑 Process
1. Data Preprocessing

Removed duplicates & handled null values.

Label Encoding → Gender

OneHot Encoding → Geography

Feature Scaling → StandardScaler

2. Model Development

Built a Neural Network using Keras.

Tuned layers, neurons, and activation functions.

Achieved ~85% accuracy on test data.

3. Streamlit App

Tab 1: Prediction

Input customer details.

Model outputs churn probability.

Example → “Customer churn probability: 72% (likely to churn).”

Tab 2: Insights

Filter data by age, gender, geography.

Charts show churn trends:

Churn higher among German customers.

Females churn more than males.

Middle-aged customers (40–60 yrs) are most at risk.

Customers with 1 product churn more than those with 2.

Inactive members churn more often.

🌍 Real-World Impact

Proactive Retention → Bank can target at-risk customers with offers.

Business Strategy → Focus campaigns on high-risk groups.

Customer Insights → Understand why people leave & improve services.

Example:

If model flags a customer with 80% churn risk, the bank can assign a retention agent.

If insights show inactive customers churn 2x more, create campaigns to increase engagement.

📸 Screenshots

(Add app screenshots → Prediction Tab & Insights Tab charts)

📂 Repository Contents

app.py → Streamlit App

model.h5 → Trained Model

scaler.pkl, label_encoder.pkl, onehot_encoder.pkl → Preprocessing objects

notebooks/ → Data cleaning & model training

dashboard_screenshots/ → Power BI/Streamlit visuals

🚀 Portfolio Value

This project shows full data science + deployment skills:

SQL/Data Prep

Machine Learning/Deep Learning

Visualization (EDA + Streamlit)

Deployment-ready app
