import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    # Use all states and loan purposes from your interface
    states_ut = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
        'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
        'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Andaman and Nicobar Islands', 'Chandigarh',
        'Dadra and Nagar Haveli and Daman and Diu', 'Lakshadweep', 'Delhi', 'Puducherry', 'Ladakh', 'Jammu and Kashmir'
    ]
    loan_purposes = ["Home Renovation", "Education", "Medical Emergency", "Wedding", "Small Business"]

    np.random.seed(42)
    n_samples = 2000

    data = pd.DataFrame({
        'age': np.random.randint(18, 85, n_samples),
        'state': np.random.choice(states_ut, size=n_samples),
        'monthly_income': np.random.randint(1000, 1000000, n_samples),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Daily Wage', 'Unemployed'], size=n_samples),
        'cibil_score': np.random.randint(300, 900, n_samples),
        'loan_amount': np.random.randint(10000, 10000000, n_samples),
        'existing_emis': np.random.randint(0, 10, n_samples),
        'loan_purpose': np.random.choice(loan_purposes, size=n_samples),
        'debt_to_income_ratio': np.round(np.random.uniform(0.1, 0.8, n_samples), 2)
    })

    # Simple approval logic for synthetic data
    data['loan_approved'] = (
        (data['cibil_score'] > 650) &
        (data['debt_to_income_ratio'] < 0.5) &
        (data['monthly_income'] > 30000) &
        (data['loan_amount'] < data['monthly_income'] * 30)
    ).astype(int)

    X = data.drop('loan_approved', axis=1)
    y = data['loan_approved']

    numeric_features = ['age', 'monthly_income', 'cibil_score', 'loan_amount', 'existing_emis', 'debt_to_income_ratio']
    categorical_features = ['state', 'employment_type', 'loan_purpose']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    model = XGBClassifier(
        objective='binary:logistic',
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train_preprocessed, y_train)

    y_pred = model.predict(X_test_preprocessed)
    y_proba = model.predict_proba(X_test_preprocessed)[:, 1]

    print("\nModel Performance Metrics:")
    print("-----------------------")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba):.3f}")

    joblib.dump(model, 'loan_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("\n✅ Model and preprocessor saved successfully!")

if __name__ == "__main__":
    main()