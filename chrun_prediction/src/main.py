from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
from sklearn.pipeline import Pipeline
from typing import List, Dict

app = FastAPI()

# Load the model and preprocessor
preprocessor = joblib.load('src/preprocessor.pkl')
model = joblib.load('src/logistic_regression_model.pkl')

class PredictRequest(BaseModel):
    data: List[Dict[str, float]]

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)

        # Preprocess the input data
        X_transformed = preprocessor.transform(df)

        # Make predictions
        predictions = model.predict(X_transformed)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(request: PredictRequest):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)

        # Preprocess the input data
        X_transformed = preprocessor.transform(df)

        # Create a SHAP explainer
        explainer = shap.Explainer(model, X_transformed)
        
        # Compute SHAP values
        shap_values = explainer(X_transformed)
        
        # Serialize SHAP values to a simple JSON-compatible format
        shap_values_dict = {
            "values": [list(values) for values in shap_values.values],
            "base_values": list(shap_values.base_values),
            "data": list(shap_values.data)
        }

        return {"shap_values": shap_values_dict}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
