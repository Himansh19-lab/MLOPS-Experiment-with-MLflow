import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix   

import seaborn as sns
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)   

# Define the model
n_estimator = 100
max_depth = 10   

mlflow.set_experiment("wine_classification_experiment")

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("n_estimators", n_estimator)
    mlflow.log_param("max_depth", max_depth)

    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)


    ## Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save plot
    plt.savefig("confusion_matrix.png")

    # Log artifact
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tag({"Author": "Himanshu", "Project": "Wine Classification"})

    # Log model
    mlflow.sklearn.log_model(model, "RanfdomForestModel")

    print(f"accuracy: {accuracy}")