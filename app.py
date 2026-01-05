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
    layout="wide",
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
        
        # EMI details for automatic DTI calculation
        st.subheader("EMI Details (for Debt-to-Income Ratio)")
        Total_EMI_Left = st.number_input("Total Outstanding EMI Amount Left (₹)", min_value=0, max_value=10000000, value=1500000, step=10000)  # Total remaining balance
        EMI_Months_Left = st.number_input("EMI Months Left", min_value=1, max_value=360, value=30, step=1)  # Must be >0 to avoid division by zero
        
        # Calculate monthly EMI and DTI automatically
        if EMI_Months_Left > 0 and MonthlyIncome > 0:
            Monthly_EMI = Total_EMI_Left / EMI_Months_Left
            Debt_to_Income_Ratio = round((Monthly_EMI / MonthlyIncome), 4)
        else:
            Monthly_EMI = 0
            Debt_to_Income_Ratio = 0.0
        
        st.write(f"**Calculated Monthly EMI: ₹{Monthly_EMI:,.0f}**")
        st.write(f"**Calculated Debt-to-Income Ratio: {Debt_to_Income_Ratio:.2f}** (Lower is better for approval)")
        
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
    
    # Debug prints
    print(f"Debug - Monthly Income: {MonthlyIncome}")
    print(f"Debug - Total EMI Left: {Total_EMI_Left}")
    print(f"Debug - EMI Months Left: {EMI_Months_Left}")
    print(f"Debug - Calculated Monthly EMI: {Monthly_EMI}")
    print(f"Debug - Calculated DTI: {Debt_to_Income_Ratio}")
    
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
    
    # Prediction Result
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
        
        # New: Empathetic Improvement Suggestions (only on rejection)
        st.markdown("""
        <div style='background: rgba(255,255,255,0.95); 
             padding: 25px;
             border-radius: 15px; 
             margin: 20px 0;
             box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #1e1e1e; margin:0 0 20px 0; font-size: 24px; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px;'>
                💡 Improvement Suggestions
            </h3>
            <p style='color: #666; margin-bottom: 20px; font-size: 16px;'>
                We understand loan rejections can be disappointing, but don't worry—many factors can be improved. Based on your details, here's personalized advice to strengthen your application for future approvals.
            </p>
        """, unsafe_allow_html=True)

        # Generative empathetic feedback based on inputs
        suggestions = []

        if Cibil_Score < 650:
            suggestions.append("**Improve Your CIBIL Score**: Your score is below 650, which is a key factor. Pay all bills on time, reduce outstanding debts, and avoid new credit applications for 6-12 months to see an increase.")

        if Debt_to_Income_Ratio > 0.5:
            suggestions.append("**Lower Your Debt-to-Income Ratio**: At {:.2f}, it's higher than ideal. Consider paying off existing EMIs faster or increasing your income to bring it under 0.4 for better approval odds.".format(Debt_to_Income_Ratio))

        if MonthlyIncome < 30000:
            suggestions.append("**Boost Your Income**: Lenders prefer stable, higher incomes. Explore side gigs, promotions, or skill-building courses to increase your monthly earnings.")

        if Loan_Amount > MonthlyIncome * 30:
            suggestions.append("**Adjust Loan Amount**: Requesting ₹{:,} is high relative to your income. Start with a smaller amount (e.g., 20-30x monthly income) to improve feasibility.".format(Loan_Amount))

        if Existing_EMIs > 2:
            suggestions.append("**Reduce Existing EMIs**: You have {} ongoing EMIs, which can strain finances. Prioritize paying off smaller loans to free up cash flow.".format(Existing_EMIs))

        if Age > 65:
            suggestions.append("**Age Considerations**: At {}, some lenders have age limits. Consider co-applicants or explore senior-friendly loan options.".format(Age))

        # Default suggestion if none apply
        if not suggestions:
            suggestions.append("**General Tips**: Review all details for accuracy. Consult a financial advisor for personalized guidance. Reapply in 3-6 months after improvements.")

        # Display suggestions
        for i, tip in enumerate(suggestions, 1):
            st.markdown(f"<p style='color: #555; font-size: 15px; margin-bottom: 15px; line-height: 1.6;'>{i}. {tip}</p>", unsafe_allow_html=True)

        st.markdown("""
            <div style='margin-top: 20px; padding: 15px; background: rgba(100,200,100,0.1); border-radius: 10px; border-left: 4px solid #28a745;'>
                <p style='margin:0; color: #1e1e1e; font-size: 14px; font-weight: 500;'>
                    🌟 Remember, rejections are not permanent. Small changes can lead to approvals. We're here to help you succeed!
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Financial Indicators
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); 
         padding: 25px;
         border-radius: 15px; 
         margin: 20px 0;
         box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3 style='color: #1e1e1e; margin:0 0 20px 0; font-size: 24px; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px;'>
            Key Financial Indicators
        </h3>
        <p style='color: #666; margin-bottom: 20px; font-size: 16px;'>
            Here's a breakdown of your financial details and what they mean for your loan application:
        </p>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div style='padding: 15px; background: rgba(240,240,240,0.3); border-radius: 10px; min-height: 120px;'>
                <p style='margin:0; color: #666; font-size: 14px;'>CIBIL Score</p>
                <h4 style='margin:5px 0 0 0; color: #1e1e1e; font-size: 20px;'>{}</h4>
                <p style='margin:10px 0 0 0; color: #555; font-size: 13px; overflow-wrap: break-word; word-wrap: break-word;'>
                    Your credit score out of 900. Higher scores (above 750) show you're a low-risk borrower and improve approval chances.
                </p>
            </div>
            <div style='padding: 15px; background: rgba(240,240,240,0.3); border-radius: 10px; min-height: 120px;'>
                <p style='margin:0; color: #666; font-size: 14px;'>Monthly Income</p>
                <h4 style='margin:5px 0 0 0; color: #1e1e1e; font-size: 20px;'>₹{:,}</h4>
                <p style='margin:10px 0 0 0; color: #555; font-size: 13px; overflow-wrap: break-word; word-wrap: break-word;'>
                    Your take-home pay each month. Lenders check if you can afford loan payments without financial strain.
                </p>
            </div>
            <div style='padding: 15px; background: rgba(240,240,240,0.3); border-radius: 10px; min-height: 120px;'>
                <p style='margin:0; color: #666; font-size: 14px;'>Loan Amount Requested</p>
                <h4 style='margin:5px 0 0 0; color: #1e1e1e; font-size: 20px;'>₹{:,}</h4>
                <p style='margin:10px 0 0 0; color: #555; font-size: 13px; overflow-wrap: break-word; word-wrap: break-word;'>
                    The money you're asking for. Banks ensure this matches your income and repayment ability.
                </p>
            </div>
            <div style='padding: 15px; background: rgba(240,240,240,0.3); border-radius: 10px; min-height: 120px;'>
                <p style='margin:0; color: #666; font-size: 14px;'>Debt-to-Income Ratio</p>
                <h4 style='margin:5px 0 0 0; color: #1e1e1e; font-size: 20px;'>{:.2f}</h4>
                <p style='margin:10px 0 0 0; color: #555; font-size: 13px; overflow-wrap: break-word; word-wrap: break-word;'>
                    How much of your income goes to debts. Lower ratios (under 0.4) mean more financial flexibility.
                </p>
            </div>
        </div>
        <div style='margin-top: 20px; padding: 15px; background: rgba(100,200,100,0.1); border-radius: 10px; border-left: 4px solid #28a745;'>
            <p style='margin:0; color: #1e1e1e; font-size: 14px; font-weight: 500; overflow-wrap: break-word; word-wrap: break-word;'>
                💡 Tip: These factors work together. A strong credit score and stable income can often outweigh a higher debt ratio.
            </p>
        </div>
    </div>
""".format(Cibil_Score, MonthlyIncome, Loan_Amount, Debt_to_Income_Ratio), unsafe_allow_html=True)

    # SHAP Analysis
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); 
         padding: 25px;
         border-radius: 15px; 
         margin: 20px 0;
         box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3 style='color: #1e1e1e; margin:0 0 20px 0; font-size: 24px; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px;'>
            Feature Impact Analysis
        </h3>
        <p style='color: #666; margin-bottom: 20px; font-size: 16px;'>
            This chart shows how each factor influenced your loan approval decision. Red bars indicate factors that decreased approval chances, while blue bars show positive influences.
        </p>
    """, unsafe_allow_html=True)

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
    st.markdown("</div>", unsafe_allow_html=True)

    # LIME Explanation
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); 
         padding: 30px;
         border-radius: 15px; 
         margin: 20px 0;
         box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3 style='color: #1e1e1e; margin:0 0 20px 0; font-size: 24px; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px;'>
            🎯 Detailed Feature Breakdown
        </h3>
        <p style='color: #666; margin-bottom: 20px; font-size: 16px;'>
            This shows how each factor specifically affected your result. Green bars boost approval chances, red bars reduce them. For rejections, most bars are red.
        </p>
    </div>
    """, unsafe_allow_html=True)

    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=preprocessor.transform(pd.DataFrame([input_dict] * 50)),
        feature_names=preprocessor.get_feature_names_out(),
        class_names=["Rejected", "Approved"],
        mode="classification"
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=data_for_shap[0],
        predict_fn=model.predict_proba,
        num_features=8
    )

    try:
        exp_list = lime_exp.as_list(label=int(prediction))
    except KeyError:
        exp_list = lime_exp.as_list()

    max_impact = max([abs(impact) for _, impact in exp_list]) if exp_list else 1.0

    lime_html = """
    <style>
        .lime-container {
            width: 100%;
            background: rgba(255,255,255,0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            box-sizing: border-box;
            margin: 0;
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            width: 100%;
        }
        .feature-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.08);
            border-left: 5px solid var(--border-color);
            box-sizing: border-box;
            min-height: 150px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .feature-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 12px rgba(0,0,0,0.12);
        }
        .feature-name {
            color: #1e1e1e;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
            margin-bottom: 10px;
        }
        .feature-direction {
            color: var(--text-color);
            font-weight: bold;
            font-size: 14px;
            text-align: center;
            margin-bottom: 15px;
        }
        .impact-bar-container {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            gap: 10px;
        }
        .impact-bar {
            height: 20px;
            border-radius: 10px;
            background: var(--bar-color);
            min-width: 5%;
            flex-grow: 1;
            max-width: 200px;
        }
        .impact-value {
            color: #666;
            font-size: 14px;
            font-weight: 500;
            white-space: nowrap;
        }
        @media (max-width: 768px) {
            .features-grid {
                grid-template-columns: 1fr;
                gap: 15px;
            }
            .feature-card {
                min-height: 120px;
                padding: 15px;
            }
            .feature-name {
                font-size: 14px;
            }
        }
    </style>
    <div class='lime-container'>
        <div class='features-grid'>
    """

    for feature, impact in exp_list:
        width_pct = max(abs(impact) / max_impact * 100 if max_impact > 0 else 0, 5)
        color = "#28a745" if impact > 0 else "#dc3545"
        direction = "Helps Approval" if impact > 0 else "Hurts Approval"
        
        lime_html += f"""
        <div class='feature-card' style='--border-color: {color}; --text-color: {color}; --bar-color: {color};'>
            <div class='feature-name'>{feature}</div>
            <div class='feature-direction'>{direction}</div>
            <div class='impact-bar-container'>
                <div class='impact-bar' style='width: {width_pct}%;'></div>
                <span class='impact-value'>{impact:.3f}</span>
            </div>
        </div>
        """

    lime_html += """
        </div>
    </div>
    """

    st.components.v1.html(lime_html, height=800, scrolling=True)

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