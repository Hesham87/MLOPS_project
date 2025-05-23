import pickle

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# Custom column dropper transformer
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns, axis=1)


# Load data
df = pd.read_csv("/home/hesham/MLOPS_project/data/raw/Titanic-Dataset.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline components
columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
numerical_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Pclass", "Sex", "Embarked"]

# Create preprocessing pipeline
preprocessing_pipeline = Pipeline(
    [
        ("drop_columns", ColumnDropper(columns_to_drop)),
        (
            "preprocessor",
            ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), numerical_features),
                    ("cat", OrdinalEncoder(), ["Sex", "Pclass"]),
                    (
                        "embarked",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="most_frequent")),
                                ("encode", OneHotEncoder()),
                            ]
                        ),
                        ["Embarked"],
                    ),
                ]
            ),
        ),
    ]
)

# Fit and transform data using pipeline
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# X_train_preprocessed.to_csv('/home/hesham/MLOPS_project/data/processed/processed_titanic_train.csv', index=False)
# X_test_preprocessed.to_csv('/home/hesham/MLOPS_project/data/processed/processed_titanic_test.csv', index=False)

# Save preprocessed data
preprocessed_train = pd.DataFrame(
    X_train_preprocessed,
    columns=numerical_features + ["Sex", "Pclass", "Embarked_C", "Embarked_Q", "Embarked_S"],
)
preprocessed_train["Survived"] = y_train.values
preprocessed_train.to_csv(
    "/home/hesham/MLOPS_project/data/processed/processed_titanic_train.csv", index=False
)

preprocessed_test = pd.DataFrame(
    X_test_preprocessed,
    columns=numerical_features + ["Sex", "Pclass", "Embarked_C", "Embarked_Q", "Embarked_S"],
)
preprocessed_test["Survived"] = y_test.values
preprocessed_test.to_csv(
    "/home/hesham/MLOPS_project/data/processed/processed_titanic_test.csv", index=False
)

# Train models on preprocessed data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

rf.fit(X_train_preprocessed, y_train)
lr.fit(X_train_preprocessed, y_train)

# Save models and preprocessing pipeline
with open("/home/hesham/MLOPS_project/models/preprocessing_pipeline.pkl", "wb") as f:
    pickle.dump(preprocessing_pipeline, f)

with open("/home/hesham/MLOPS_project/models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("/home/hesham/MLOPS_project/models/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(lr, f)

print("Preprocessed data and models saved successfully")
