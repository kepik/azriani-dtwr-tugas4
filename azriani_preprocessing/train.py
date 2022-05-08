import pandas as pd
import pickle # joblib

from helper.data_check_preparation import read_and_check_data
from helper.feature_engineering import feature_engineering
from helper.constant import TRAIN_COLUMN, PATH
from helper.models import LINEAR_MODEL_CLF
from helper.preprocessing import standard_scaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def train_model():
    # pembacaan dan pengecekan data
    df = read_and_check_data(PATH, TRAIN_COLUMN)
    
    # feature engineering
    df_transformed = feature_engineering(df)
    print("Start Saving Result Feature Engineering!")
    df_transformed.to_csv("artifacts/df_transformed.csv")
    
    # siapkan fitur data dan target data 
    X = df_transformed.drop(["price"], axis =1)
    y = df_transformed["price"]
    
    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    # scaler
    train_scaler, X_train_scaled = standard_scaler(X_train)
    pickle.dump(train_scaler, open("artifacts/train_scaler.pkl", "wb"))
    X_test_scaled = train_scaler.transform(X_test)
    
    print("Start Saving Result Standard Scaler!")
    X_train_scaled.to_csv("artifacts/X_train_scaled.csv")
    
    # define model and start training
    clf_model = LINEAR_MODEL_CLF["logreg_cv"]
    clf_model.fit(X_train_scaled, y_train)
    y_pred_class = clf_model.predict_proba(X_test_scaled).argmax(1)
    y_pred_proba = clf_model.predict_proba(X_test_scaled)[:, 1]

    pickle.dump(clf_model, open("artifacts/logreg_model.pkl", "wb"))
    
    # show training result
    print("------------------------------")
    print("Model Performance:")
    print("ROC_AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Recall:", recall_score(y_test, y_pred_class))
    print("Precision:", precision_score(y_test, y_pred_class))
    print("f1_score:", f1_score(y_test, y_pred_class, average="macro")) 
    
if __name__ == "__main__":
    print("START RUNNING PIPELINE!")
    train_model()
    print("FINISH RUNNING PIPELINE!")
    