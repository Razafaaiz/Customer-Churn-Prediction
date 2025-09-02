import streamlit as st
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# =====================
# Load Model & Encoders
# =====================
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load dataset (for insights page)
df = pd.read_csv("Churn_Modelling.csv")  # Replace with your dataset

# =====================
# Streamlit Styling
# =====================
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #FAFAFA; }
    .stCard {
        background: #1c1f26; padding: 20px; border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================
# Sidebar Info
# =====================
st.sidebar.title("ℹ️ About this App")
st.sidebar.write(
    """
    **This app is a predictive analytics tool that acts as a customer risk scoring system.**  
    - Built with: TensorFlow, Streamlit  
    - Dataset: Bank Customer Churn Dataset  
    - Goal: Predict whether a customer will leave (churn).  
    """
)
st.sidebar.write("👨‍💻 Developer: FAIZ RAZA")

# =====================
# Tabs
# =====================
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Insights"])

# =====================
# TAB 1: Prediction
# =====================
with tab1:
    st.title("🔮 Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 30)
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650)

    with col2:
        balance = st.number_input('🏦 Balance', min_value=0.0, value=50000.0)
        estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=60000.0)
        tenure = st.slider('📆 Tenure (Years with Bank)', 0, 10, 5)
        num_of_products = st.slider('🛒 Number of Products', 1, 4, 1)
        has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
        is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

    # Prepare input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.markdown("### 📊 Prediction Result")
    st.progress(float(prediction_proba))

    if prediction_proba > 0.5:
        st.error(f"⚠️ Likely to churn. Probability: {prediction_proba:.2%}")
    else:
        st.success(f"✅ Not likely to churn. Probability: {prediction_proba:.2%}")

    with st.expander("🔍 Why this prediction?"):
        st.write("We did this churn prediction to help businesses perform customer retention analysis and protect revenue. By identifying high-risk customers early, companies can take proactive, data-driven actions—like loyalty offers or targeted support—that improve customer lifetime value and reduce churn.")

# =====================
# TAB 2: Insights (Interactive)
# =====================
with tab2:
    st.title("📊 Customer Churn Insights")

    # ---- Sidebar Filters ----
    st.sidebar.subheader("🔎 Insights Filters")

    geo_filter = st.sidebar.multiselect("🌍 Select Geography", df['Geography'].unique(), default=list(df['Geography'].unique()))
    gender_filter = st.sidebar.multiselect("👤 Select Gender", df['Gender'].unique(), default=list(df['Gender'].unique()))
    age_range = st.sidebar.slider("🎂 Age Range", int(df['Age'].min()), int(df['Age'].max()), (20, 60))
    product_filter = st.sidebar.multiselect("🛒 Number of Products", sorted(df['NumOfProducts'].unique()), default=sorted(df['NumOfProducts'].unique()))

    # ---- Apply Filters ----
    df_filtered = df[
        (df['Geography'].isin(geo_filter)) &
        (df['Gender'].isin(gender_filter)) &
        (df['Age'].between(age_range[0], age_range[1])) &
        (df['NumOfProducts'].isin(product_filter))
    ]

    # ---- Metrics ----
    churn_rate = df_filtered['Exited'].mean() * 100 if len(df_filtered) > 0 else 0
    st.metric("Overall Churn Rate (Filtered)", f"{churn_rate:.1f}%")

    # ---- Charts ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Gender")
        st.bar_chart(df_filtered.groupby('Gender')['Exited'].mean())

    with col2:
        st.subheader("Churn by Geography")
        st.bar_chart(df_filtered.groupby('Geography')['Exited'].mean())

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn by Age Group")
        df_filtered['AgeGroup'] = pd.cut(df_filtered['Age'], bins=[18,30,40,50,60,92], labels=['18-30','31-40','41-50','51-60','61+'])
        st.bar_chart(df_filtered.groupby('AgeGroup')['Exited'].mean())

    with col4:
        st.subheader("Churn by Products Owned")
        st.bar_chart(df_filtered.groupby('NumOfProducts')['Exited'].mean())


