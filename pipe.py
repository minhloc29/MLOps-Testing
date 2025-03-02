import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import zenml
from zenml import step, pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

@step 
def ingest_data(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file)

@step
def load_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df.dropna(inplace=True)
    X = df.drop(columns=['class'])
    Y = df['class']
    return X, Y

@step 
def preprocessing(x, y) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.1, stratify=y_encoded, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )  
    return X_train, y_train, X_val, y_val, X_test, y_test

@step 
def train_model(x, y) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x, y)
    return model

@step
def evaluator(model: KNeighborsClassifier, x_val, y_val) -> str:
    y_pred = model.predict(x_val)
    val_precision = precision_score(y_val, y_pred, average='macro')  # 'micro', 'weighted' are other options
    val_recall = recall_score(y_val, y_pred, average='macro')
    val_f1 = f1_score(y_val, y_pred, average='macro')  # Options: 'micro', 'weighted'  
    return f"Precision: {val_precision}\nRecall: {val_recall}\nF1-Score: {val_f1}"

@pipeline(enable_cache=True) #if nothing changes in steps, use previous version, reduce time loaded
def my_pipeline():
    df = ingest_data("vehicle.csv")
    X, Y = load_data(df)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing(X, Y)
    model = train_model(X_train, y_train)
    eval_score = evaluator(model, X_val, y_val)
    return eval_score

eval_sce = my_pipeline()
print(eval_sce)