import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd

def load_model():
    # Load preprocessor and model
    preprocessor = joblib.load('src/preprocessor.pkl')
    model = joblib.load('src/logistic_regression_model.pkl')
    return model, preprocessor

def preprocess_data(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    # Separate features from the target if present
    if 'churn' in df.columns:
        X = df.drop(columns=['churn'])
    else:
        X = df
    return preprocessor.transform(X)

def get_model_predictions(model: Pipeline, X_transformed: pd.DataFrame) -> pd.Series:
    return model.predict(X_transformed)
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def get_feature_transformation_pipeline(df):
    # Separate target variable 'churn' from features
    X = df.drop(columns=['churn'])
    y = df['churn']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing for numerical features: Imputation + Scaling
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())  # Scale numerical features
    ])

    # Preprocessing for categorical features: Imputation + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical features
    ])

    # Combine transformations using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Return the preprocessor pipeline
    return preprocessor, X, y

def train_and_save_model():
    
    # Load the dataset
    df = pd.read_csv(r'E:\survey sparrow\chrun_prediction\data\Customertravel.csv')


    # Get the transformation pipeline
    preprocessor, X, y = get_feature_transformation_pipeline(df)

    # Transform the features
    X_transformed = preprocessor.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

    # Define the model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Save the model and preprocessing pipeline
    joblib.dump(preprocessor, 'src/preprocessor.pkl')
    joblib.dump(model, 'src/logistic_regression_model.pkl')

    print("Model and preprocessor saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
