import shap
import joblib
import pandas as pd

def explain_model(X: pd.DataFrame, model: joblib.Parallel) -> shap.Explanation:
    
    # Load the preprocessing pipeline and model
    preprocessor = joblib.load('src/preprocessor.pkl')
    model = joblib.load('src/logistic_regression_model.pkl')

    # Transform the input features
    X_transformed = preprocessor.transform(X)

    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_transformed)
    
    # Compute SHAP values
    shap_values = explainer(X_transformed)
    
    return shap_values



if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv(r'E:\survey sparrow\chrun_prediction\data\Customertravel.csv')

    X = df.drop(columns=['churn'])

    # Generate explanations for the first few rows
    shap_values = explain_model(X.head())
    
    # Plot SHAP values
    shap.summary_plot(shap_values, X)