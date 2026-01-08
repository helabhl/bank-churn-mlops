import streamlit as st
import requests
import pandas as pd
import json
import os

# Configuration
API_URL = os.getenv("API_URL", "https://churn-api.jollybush-dda26c9c.swedencentral.azurecontainerapps.io")

st.set_page_config(
    page_title="Bank Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Churn Prediction & Monitoring")

# Sidebar for configuration
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", API_URL)

# Check API Health
try:
    health = requests.get(f"{api_url}/health")
    if health.status_code == 200:
        st.sidebar.success("API Connected ✅")
    else:
        st.sidebar.error(f"API Error: {health.status_code}")
except Exception:
    st.sidebar.error("API Unreachable ❌")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction", "📉 Drift Detection"])

# ============================================================
# TAB 1: SINGLE PREDICTION
# ============================================================
with tab1:
    st.header("Predict Churn for a Single Client")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.number_input("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
        estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 75000.0)

    with col2:
        num_of_products = st.number_input("Number of Products", 1, 4, 2)
        has_cr_card = st.selectbox("Has Credit Card?", [0, 1], index=1)
        is_active_member = st.selectbox("Is Active Member?", [0, 1], index=1)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    # Convert Geography to dummy variables
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    if st.button("Predict"):
        payload = {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "Geography_Germany": geo_germany,
            "Geography_Spain": geo_spain
        }

        try:
            response = requests.post(f"{api_url}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                
                st.subheader("Prediction Result")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Churn Probability", f"{result['churn_probability'] * 100:.2f}%")
                col_res2.metric("Prediction", "Churn" if result['prediction'] == 1 else "Stay")
                col_res3.metric("Risk Level", result['risk_level'])

                if result['prediction'] == 1:
                    st.error("⚠️ This customer is likely to churn.")
                else:
                    st.success("✅ This customer is likely to stay.")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {str(e)}")

# ============================================================
# TAB 2: BATCH PREDICTION
# ============================================================
with tab2:
    st.header("Batch Prediction via CSV")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Predict Batch"):
            try:
                # Prepare payload
                # Note: We need to ensure the columns match the expected input
                # For simplicity, we assume the CSV has the raw columns and we process them if needed
                # However, the API expects specific feature structure.
                # Let's map or validate. For this demo, we assume the CSV is pre-processed OR 
                # we need to transform 'Geography' if present, etc.
                
                # To be robust, let's assume the CSV structure matches the training data slightly.
                # If 'Geography' column exists, we dummies it.
                
                df_processed = df.copy()
                if 'Geography' in df_processed.columns:
                    df_processed['Geography_Germany'] = (df_processed['Geography'] == 'Germany').astype(int)
                    df_processed['Geography_Spain'] = (df_processed['Geography'] == 'Spain').astype(int)
                    if 'Geography_France' in df_processed.columns: # Clean up if present from dummies
                        pass 
                
                # Verify columns exist
                required_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                                 "HasCrCard", "IsActiveMember", "EstimatedSalary", 
                                 "Geography_Germany", "Geography_Spain"]
                
                missing = [c for c in required_cols if c not in df_processed.columns]
                
                if missing:
                    st.error(f"Missing columns in CSV: {missing}")
                    st.info("Ensure you have 'Geography' to convert, or 'Geography_Germany'/'Geography_Spain' directly.")
                else:
                    records = df_processed[required_cols].to_dict(orient="records")
                    
                    response = requests.post(f"{api_url}/predict/batch", json=records)
                    
                    if response.status_code == 200:
                        results = response.json()["predictions"]
                        result_df = pd.DataFrame(results)
                        
                        # Combine with original
                        final_df = pd.concat([df, result_df], axis=1)
                        st.success(f"Processed {len(results)} records.")
                        st.dataframe(final_df)
                        
                        # Download button
                        csv = final_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Results", csv, "predictions.csv", "text/csv")
                    else:
                        st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Batch processing failed: {str(e)}")

# ============================================================
# TAB 3: DRIFT DETECTION
# ============================================================
with tab3:
    st.header("Drift Detection")
    st.write("Compare production data against reference training data.")

    threshold = st.slider("Drift Threshold (p-value)", 0.01, 0.10, 0.05, 0.01)

    if st.button("Check Drift"):
        try:
            response = requests.post(f"{api_url}/drift/check?threshold={threshold}")
            if response.status_code == 200:
                data = response.json()
                
                st.metric("Features Analyzed", data['features_analyzed'])
                st.metric("Features with Drift", data['features_drifted'])
                
                if data['features_drifted'] > 0:
                    st.warning("⚠️ Data Drift Detected!")
                else:
                    st.success("✅ No Data Drift Detected.")
                    
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Drift check failed: {str(e)}")
