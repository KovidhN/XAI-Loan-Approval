import streamlit as st
import pandas as pd
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

# ----- PAGE CONFIG -----
st.set_page_config(
    page_title="AI Loan Approval Pro",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -- Load Model & Preprocessor --
model = joblib.load('loan_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

states_ut = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
    'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
    'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Andaman and Nicobar Islands', 'Chandigarh',
    'Dadra and Nagar Haveli and Daman and Diu', 'Lakshadweep', 'Delhi', 'Puducherry', 'Ladakh', 'Jammu and Kashmir'
]

st.markdown("""
<div style='background: linear-gradient(90deg, #0052D4, #4364F7, #6FB1FC); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; text-align: center;'>AI Loan Approval Predictor</h1>
    <p style='color: #e0e0e0; text-align: center;'>India-Ready | Explainable AI | Instant Results</p>
</div>
""", unsafe_allow_html=True)

with st.form("input_form"):
    st.header("Applicant Information")
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=85, value=30)
        State = st.selectbox("State/UT", states_ut, index=12)
        MonthlyIncome = st.number_input("Monthly Income (₹)", min_value=1000, max_value=1000000, value=60000)
        EmploymentType = st.selectbox("Employment Type ⓘ", ["Salaried", "Self-Employed", "Daily Wage", "Unemployed"], index=0)
    with col2:
        Cibil_Score = st.slider("CIBIL Score", min_value=300, max_value=900, value=750)
        Loan_Amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=800000, step=5000)
        Existing_EMIs = st.number_input("Existing EMIs", min_value=0, max_value=10, value=1)
        Loan_Purpose = st.selectbox("Loan Purpose", ["Home Renovation", "Education", "Medical Emergency", "Wedding", "Small Business"], index=0)
        Debt_to_Income_Ratio = st.slider("Debt-to-Income Ratio ⓘ", min_value=0.10, max_value=0.80, value=0.32, step=0.01)
    submit = st.form_submit_button("Predict Now")

if submit:
    input_dict = {
        "age": Age,
        "state": State,
        "monthly_income": MonthlyIncome,
        "employment_type": EmploymentType,
        "cibil_score": Cibil_Score,
        "loan_amount": Loan_Amount,
        "existing_emis": Existing_EMIs,
        "loan_purpose": Loan_Purpose,
        "debt_to_income_ratio": Debt_to_Income_Ratio
    }
    input_df = pd.DataFrame([input_dict])

    # Handle unseen categorical values
    cat_columns = ['loan_purpose', 'employment_type', 'state']
    for col in cat_columns:
        try:
            cat_transformer = preprocessor.named_transformers_['cat']
            feature_names = preprocessor.transformers_[1][2]
            if col in feature_names:
                feature_index = feature_names.index(col)
                known_values = cat_transformer.categories_[feature_index]
                if input_df[col].iloc[0] not in known_values:
                    st.warning(f"'{input_df[col].iloc[0]}' is an unknown category for '{col}'. Replacing with 'Other'.")
                    input_df[col] = 'Other'
        except Exception as e:
            st.error(f"Category handling failed for '{col}': {e}")

    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]
    prediction_proba = model.predict_proba(processed_input)[0]

    st.header("Prediction Result")
    if prediction == 1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
             color: white; 
             padding: 30px;
             border-radius: 15px; 
             margin: 20px 0;
             box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h2 style='margin:0; text-align:center; font-size: 28px;'>✅ Loan Approved!</h2>
            <p style='margin:15px 0 0 0; text-align:center; font-size: 18px;'>
                Confidence Score: {:.1f}%
            </p>
        </div>
    """.format(prediction_proba[1]*100), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); 
             color: white; 
             padding: 30px;
             border-radius: 15px; 
             margin: 20px 0;
             box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h2 style='margin:0; text-align:center; font-size: 28px;'>❌ Loan Not Approved</h2>
            <p style='margin:15px 0 0 0; text-align:center; font-size: 18px;'>
                Confidence Score: {:.1f}%
            </p>
        </div>
    """.format(prediction_proba[0]*100), unsafe_allow_html=True)

    # Enhanced Summary Section
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); 
         padding: 25px;
         border-radius: 15px; 
         margin: 20px 0;
         box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3 style='color: #1e1e1e; margin:0 0 20px 0; font-size: 24px; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px;'>
            Key Financial Indicators
        </h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div style='padding: 15px; background: rgba(240,240,240,0.3); border-radius: 10px;'>
                <p style='margin:0; color: #666; font-size: 14px;'>CIBIL Score</p>
                <h4 style='margin:5px 0 0 0; color: #1e1e1e; font-size: 20px;'>{}</h4>
            </div>
            <div style='padding: 15px; background: rgba(240,240,240,0.3); border-radius: 10px;'>
                <p style='margin:0; color: #666; font-size: 14px;'>Monthly Income</p>
                <h4 style='margin:5px 0 0 0; color: #1e1e1e; font-size: 20px;'>₹{:,}</h4>
            </div>
            <div style='padding: 15px; background: rgba(240,240,240,0.3); border-radius: 10px;'>
                <p style='margin:0; color: #666; font-size: 14px;'>Loan Amount</p>
                <h4 style='margin:5px 0 0 0; color: #1e1e1e; font-size: 20px;'>₹{:,}</h4>
            </div>
            <div style='padding: 15px; background: rgba(240,240,240,0.3); border-radius: 10px;'>
                <p style='margin:0; color: #666; font-size: 14px;'>Debt-to-Income Ratio</p>
                <h4 style='margin:5px 0 0 0; color: #1e1e1e; font-size: 20px;'>{:.2f}</h4>
            </div>
        </div>
    </div>
""".format(Cibil_Score, MonthlyIncome, Loan_Amount, Debt_to_Income_Ratio), unsafe_allow_html=True)

    # SHAP Analysis with better styling
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); 
         padding: 25px;
         border-radius: 15px; 
         margin: 20px 0;
         box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3 style='color: #1e1e1e; margin:0 0 20px 0; font-size: 24px; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px;'>
            Feature Impact Analysis
        </h3>
    </div>
""", unsafe_allow_html=True)

    # SHAP plot
    if hasattr(processed_input, "toarray"):
        data_for_shap = processed_input.toarray()
    else:
        data_for_shap = processed_input

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_for_shap)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        data_for_shap,
        feature_names=preprocessor.get_feature_names_out(),
        show=False
    )
    st.pyplot(fig)

    # LIME Explanation with error handling
    st.markdown("### 🎯 Feature-by-Feature Analysis")
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=preprocessor.transform(pd.DataFrame([input_dict] * 50)),
        feature_names=preprocessor.get_feature_names_out(),
        class_names=["Rejected", "Approved"],
        mode="classification"
    )

    # Generate explanation with proper label handling
    lime_exp = lime_explainer.explain_instance(
        data_row=data_for_shap[0],
        predict_fn=model.predict_proba,
        num_features=8
    )

    # Get the explanation for the predicted class
    exp_list = lime_exp.as_list(label=int(prediction))

    # Create custom HTML for LIME visualization with error handling
    max_impact = max([abs(impact) for _, impact in exp_list], default=1.0)  # Use default value if empty

    lime_html = """
    <div style='background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3 style='margin-bottom: 20px; color: #1e1e1e;'>Impact of Features on Prediction</h3>
    """

    for feature, impact in exp_list:
        try:
            # Calculate width percentage with safety check
            width_pct = min(abs(impact) / max_impact * 100, 100) if max_impact != 0 else 0
            # Determine color based on impact
            color = "#28a745" if impact > 0 else "#dc3545"
            
            lime_html += f"""
            <div style='margin: 10px 0; padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <span style='color: #1e1e1e; font-weight: 500;'>{feature}</span>
                    <span style='color: #666;'>{impact:.3f}</span>
                </div>
                <div style='height: 8px; border-radius: 4px; background: {color}; width: {width_pct}%;'></div>
            </div>
            """
        except Exception as e:
            st.error(f"Error processing feature {feature}: {str(e)}")

    lime_html += "</div>"

    # Display the custom LIME visualization
    st.components.v1.html(lime_html, height=600, scrolling=True)

# -- Background styling --
st.markdown(
    """ <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1565372918675-6d0d4a4f27c8?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    } </style>
    """,
    unsafe_allow_html=True
)