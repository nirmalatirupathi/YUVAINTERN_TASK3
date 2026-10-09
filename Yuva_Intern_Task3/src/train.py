"""Task 3 ML training template. Add an approved dataset before execution."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
DATA_PATH=Path("data/raw/customer_churn.csv")
TARGET_COLUMN="churn"
def preprocessor(X):
    n=X.select_dtypes(include=["number"]).columns; c=X.select_dtypes(exclude=["number"]).columns
    return ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),n),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),c)])
def main():
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df=pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns: raise ValueError(f"Missing target: {TARGET_COLUMN}")
    X=df.drop(columns=[TARGET_COLUMN]); y=df[TARGET_COLUMN]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    models={"logistic_regression":LogisticRegression(max_iter=1000),"decision_tree":DecisionTreeClassifier(random_state=42),"random_forest":RandomForestClassifier(n_estimators=200,random_state=42),"gradient_boosting":GradientBoostingClassifier(random_state=42)}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for name,m in models.items():
        pipe=Pipeline([("preprocess",preprocessor(Xtr)),("model",m)])
        s=cross_validate(pipe,Xtr,ytr,cv=cv,scoring=["accuracy","precision","recall","f1","roc_auc"])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
        print(name,{k:float(s["test_"+k].mean()) for k in ["accuracy","precision","recall","f1","roc_auc"]})
        print("test",accuracy_score(yte,pred),precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0),f1_score(yte,pred,zero_division=0),roc_auc_score(yte,prob),confusion_matrix(yte,pred).tolist())
if __name__=="__main__": main()
