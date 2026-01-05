import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from lightgbm import LGBMClassifier  # Better model for handling overfitting
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    states_ut = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
        'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
        'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Andaman and Nicobar Islands', 'Chandigarh',
        'Dadra and Nagar Haveli and Daman and Diu', 'Lakshadweep', 'Delhi', 'Puducherry', 'Ladakh', 'Jammu and Kashmir'
    ]
    loan_purposes = ["Home Renovation", "Education", "Medical Emergency", "Wedding", "Small Business"]

    np.random.seed(42)
    n_samples = 20000  # More data for better training

    data = pd.DataFrame({
        'age': np.random.randint(18, 85, n_samples),
        'state': np.random.choice(states_ut, size=n_samples),
        'monthly_income': np.random.randint(1000, 1000000, n_samples),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Daily Wage', 'Unemployed'], size=n_samples),
        'cibil_score': np.random.randint(300, 900, n_samples),
        'loan_amount': np.random.randint(10000, 10000000, n_samples),
        'existing_emis': np.random.randint(0, 10, n_samples),
        'loan_purpose': np.random.choice(loan_purposes, size=n_samples),
        # Calculate DTI realistically based on EMIs
        'emi_amount': np.random.randint(0, 50000, n_samples),
        'emi_months_left': np.random.randint(0, 360, n_samples),
    })
    
    # Calculate DTI as in the app
    data['debt_to_income_ratio'] = np.where(
        data['monthly_income'] > 0,
        data['emi_amount'] / data['monthly_income'],
        0.0
    )
    
    # Approval logic emphasizing DTI
    data['loan_approved'] = (
        (data['cibil_score'] > 650) &
        (data['debt_to_income_ratio'] < 0.4) &  # Stronger DTI check
        (data['monthly_income'] > 25000) &
        (data['loan_amount'] < data['monthly_income'] * 40) &
        (data['existing_emis'] < 4) &
        (data['age'] < 70)
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

    # Use LightGBM for better performance and less overfitting
    model = LGBMClassifier(
        objective='binary',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42
    )

    # Hyperparameter tuning
    param_grid = {
        'num_leaves': [20, 31, 40],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_preprocessed, y_train)
    model = grid_search.best_estimator_

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_preprocessed, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    y_pred = model.predict(X_test_preprocessed)
    y_proba = model.predict_proba(X_test_preprocessed)[:, 1]

    print("\nModel Performance Metrics:")
    print("-----------------------")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba):.3f}")

    # Overfitting check
    train_accuracy = accuracy_score(y_train, model.predict(X_train_preprocessed))
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTrain Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    if train_accuracy - test_accuracy > 0.05:
        print("⚠️ Possible overfitting. Consider more data or regularization.")
    else:
        print("✅ No significant overfitting.")

    # Feature importance check
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        for name, imp in zip(feature_names, importances):
            if 'debt_to_income' in name:
                print(f"DTI Feature Importance: {imp:.4f}")

    joblib.dump(model, 'loan_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("\n✅ Improved model saved!")

if __name__ == "__main__":
    main()