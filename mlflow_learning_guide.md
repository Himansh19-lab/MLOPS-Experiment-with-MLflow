# MLflow — Learn It From First Principles

## 1. What problem does MLflow solve?

Imagine you train 20 versions of a model.

For each experiment, you change things like:

- Learning rate
- Number of trees
- Features
- Dataset version
- Model type

After a few days, you might have:

```text
experiment_1 → accuracy 0.81
experiment_2 → accuracy 0.84
experiment_3 → accuracy 0.79
...
experiment_20 → accuracy 0.91
```

Now someone asks:

> Which model produced 0.91, with which parameters, on which data, and where is the model?

Without proper tracking, this becomes painful.

**MLflow helps you manage the lifecycle of ML experiments and models.**

A useful mental model is:

```text
Code + Data + Parameters
          ↓
       Training
          ↓
       MLflow
     ↙    ↓    ↘
 Metrics  Params  Artifacts
          ↓
       Model
          ↓
    Model Registry
          ↓
      Deployment
          ↓
     Monitoring
```

---

## 2. What is MLflow?

### Simple definition

**MLflow is an open-source platform for managing the machine-learning lifecycle.**

### Technical perspective

MLflow provides tools for:

1. Experiment tracking
2. Model packaging
3. Model registry
4. Model deployment
5. Evaluation and related ML lifecycle workflows

The most important beginner concept is **experiment tracking**.

---

## 3. Why was MLflow needed?

Before experiment-tracking tools, teams often did something like:

```python
accuracy = 0.91
learning_rate = 0.01
```

and then manually recorded results in:

```text
Excel
Google Sheets
Notion
text files
```

Or worse:

```text
model_final.pkl
model_final_v2.pkl
model_final_really_final.pkl
model_final_really_final_v2.pkl
```

The fundamental problem is:

> **ML experiments generate many pieces of information that need to remain connected.**

For one training run, we may have:

```text
Parameters
Metrics
Dataset information
Source code
Model
Plots
Logs
Environment
```

MLflow gives us a structured way to track these things.

---

## 4. The most important concept: Run

Think of an MLflow **run** as:

> **One record of one experiment/training attempt.**

For example:

```text
Run #1
--------------
Model: RandomForest
n_estimators: 100
max_depth: 5
accuracy: 0.82
```

Another:

```text
Run #2
--------------
Model: RandomForest
n_estimators: 300
max_depth: 10
accuracy: 0.89
```

Another:

```text
Run #3
--------------
Model: XGBoost
learning_rate: 0.05
max_depth: 6
accuracy: 0.92
```

MLflow lets you compare these runs.

---

## 5. Parameters vs Metrics

This distinction is extremely important.

### Parameters

Parameters are values you **choose before or during training**.

Examples:

```text
learning_rate = 0.01
max_depth = 5
n_estimators = 100
batch_size = 32
```

In MLflow:

```python
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("max_depth", 5)
```

---

### Metrics

Metrics are values you **measure as a result of training/evaluation**.

Examples:

```text
accuracy = 0.92
precision = 0.89
recall = 0.91
rmse = 2.31
```

MLflow:

```python
mlflow.log_metric("accuracy", 0.92)
mlflow.log_metric("precision", 0.89)
```

### Mental shortcut

```text
PARAMETER → What did I choose?

METRIC    → What did I get?
```

---

## 6. Artifacts

Not everything is a number or string.

Training may produce:

```text
model.pkl
confusion_matrix.png
feature_importance.png
classification_report.txt
requirements.txt
```

These are **artifacts**.

Example:

```python
mlflow.log_artifact("confusion_matrix.png")
```

So a run might contain:

```text
Run
│
├── Parameters
│   ├── learning_rate
│   └── max_depth
│
├── Metrics
│   ├── accuracy
│   └── f1_score
│
└── Artifacts
    ├── model.pkl
    └── confusion_matrix.png
```

---

## 7. Experiment

An **experiment** is a logical grouping of related runs.

For example:

```text
Experiment: Customer Churn Prediction

    Run 1 → Logistic Regression
    Run 2 → Random Forest
    Run 3 → XGBoost
    Run 4 → XGBoost tuned
    Run 5 → XGBoost + feature engineering
```

This is much easier to reason about than having hundreds of unrelated runs.

---

## 8. MLflow workflow

A simple MLflow workflow looks like this:

```text
Create Experiment
       ↓
Start Run
       ↓
Log Parameters
       ↓
Train Model
       ↓
Log Metrics
       ↓
Log Artifacts
       ↓
Log Model
       ↓
Compare Runs
```

---

## 9. Installation

Typical installation:

```bash
pip install mlflow
```

Then verify:

```bash
mlflow --version
```

Because MLflow evolves, check the current official documentation when working with a specific MLflow version or production setup.

---

## 10. First MLflow example

Let's use Scikit-learn.

```python
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load data
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start MLflow run
with mlflow.start_run():

    # Model configuration
    n_estimators = 100
    max_depth = 5

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model"
    )
```

The important pattern is:

```python
with mlflow.start_run():

    mlflow.log_param(...)
    mlflow.log_metric(...)
    mlflow.log_artifact(...)
```

---

## 11. What actually happened?

When this executes:

```python
with mlflow.start_run():
```

MLflow creates a **run**.

Then:

```python
mlflow.log_param(...)
```

stores configuration.

Then:

```python
mlflow.log_metric(...)
```

stores evaluation results.

Then:

```python
mlflow.sklearn.log_model(...)
```

stores the trained model.

Conceptually:

```text
                  MLflow Run
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
   Parameters     Metrics       Artifacts
       │             │             │
       ↓             ↓             ↓
 learning_rate    accuracy       model
 max_depth        F1             plots
```

---

## 12. MLflow UI

One of MLflow's biggest advantages is being able to inspect experiments through its UI.

A common local workflow is:

```bash
mlflow server
```

Then open the local MLflow interface in your browser.

The UI allows you to inspect and compare runs rather than manually reading logs.

---

## 13. Why this matters in real projects

Suppose you have:

```text
Model A
accuracy = 0.88

Model B
accuracy = 0.91

Model C
accuracy = 0.94
```

You choose Model C.

Six months later someone asks:

> Why did we choose Model C?

You want to be able to answer:

```text
Model C
│
├── Dataset version
├── Features
├── Hyperparameters
├── Training code/version
├── Metrics
├── Model artifact
└── Experiment history
```

This is the beginning of **reproducible ML engineering**.

---

## 14. MLflow architecture

A simplified architecture:

```text
                  ML Training Code
                         │
                         ↓
                    MLflow Client
                         │
                         ↓
                 MLflow Tracking
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
       Backend Store          Artifact Store
              │                     │
              ↓                     ↓
      Run metadata             Models
      Parameters               Images
      Metrics                  Files
```

The **backend store** holds tracking metadata.

The **artifact store** holds larger files such as:

```text
models
plots
datasets/artifacts
reports
```

In production, these may be backed by databases and object storage rather than local files.

---

## 15. MLflow's major components

Think of MLflow as a toolbox rather than one feature.

### 1. Tracking

Records:

```text
parameters
metrics
artifacts
runs
experiments
```

### 2. Models

Provides standardized ways to save and work with models.

### 3. Model Registry

Helps manage model versions and their lifecycle.

Conceptually:

```text
Model
 ↓
Version 1
Version 2
Version 3
```

### 4. Deployment

Provides mechanisms for serving/deploying models.

---

## 16. Tracking vs Model Registry

This is a common interview question.

### Tracking

Answers:

> **"What happened during my experiments?"**

Example:

```text
Run 42
accuracy = 0.93
learning_rate = 0.01
```

### Model Registry

Answers:

> **"Which model version are we managing/deploying?"**

Example:

```text
FraudModel
 ├── Version 1
 ├── Version 2
 └── Version 3
```

### Easy mental model

```text
Tracking
   ↓
Experiments

Registry
   ↓
Models
```

---

## 17. A realistic MLOps pipeline

Connect MLflow to the broader MLOps lifecycle:

```text
             Data
               ↓
       Data Validation
               ↓
      Feature Engineering
               ↓
           Training
               ↓
        ┌──────────────┐
        │    MLflow    │
        │   Tracking   │
        └──────────────┘
               ↓
       Model Evaluation
               ↓
        Model Registry
               ↓
          Deployment
               ↓
          Monitoring
               ↓
       Drift Detection
               ↓
          Retraining
               └───────────────→ Training
```

MLflow is **not the entire MLOps platform**.

It can be an important component within the larger system.

---

## 18. Common misconception

### ❌ "MLflow trains my model."

Not primarily.

Your framework trains the model:

```text
Scikit-learn
PyTorch
TensorFlow
XGBoost
...
```

MLflow helps manage what happens around that training.

Think:

```text
PyTorch / sklearn
       ↓
    TRAINING

MLflow
       ↓
TRACKING + MODEL LIFECYCLE
```

---

## 19. When should you use MLflow?

### Good use cases

Use MLflow when:

- You have multiple experiments.
- You need experiment reproducibility.
- You want centralized experiment tracking.
- Multiple people work on ML projects.
- You need model version management.
- You are building an MLOps workflow.

### You may not need it yet

For a tiny learning exercise:

```python
model.fit(X, y)
```

MLflow may add unnecessary complexity.

A useful engineering principle:

> **Use infrastructure when the problem justifies the infrastructure.**

---

## 20. Interview questions

### Beginner

1. What is MLflow?
2. Why is MLflow used?
3. What is an MLflow run?
4. What is an experiment?
5. What is a parameter?
6. What is a metric?
7. What is an artifact?
8. What is experiment tracking?
9. Why is reproducibility important?
10. What problem does MLflow solve?

### Intermediate

11. Tracking vs Model Registry?
12. Backend store vs artifact store?
13. How would you integrate MLflow into a training pipeline?
14. How do you log a model?
15. How would you compare multiple experiments?
16. How would you use MLflow in CI/CD?
17. Where does MLflow fit into MLOps?
18. How would you deploy an MLflow-tracked model?
19. How would you manage model versions?
20. How would you make MLflow production-ready?

### Advanced

21. Design a centralized MLflow architecture for multiple teams.
22. How would you secure an MLflow server?
23. How would you manage artifact storage at scale?
24. How would you integrate MLflow with cloud object storage?
25. How would you implement model promotion?
26. How would you handle rollback?
27. How would MLflow integrate with Kubernetes?
28. How would you manage experiment reproducibility?
29. How would you handle model lineage?
30. How would you integrate MLflow into a full CI/CD pipeline?

---

## 21. A strong interview answer

If an interviewer asks:

> **What is MLflow?**

A good answer:

> "MLflow is an open-source platform for managing the machine-learning lifecycle. One of its core capabilities is experiment tracking, where we can record parameters, metrics, artifacts, and models for individual training runs. It also provides model-management and deployment capabilities, including a model registry. In an MLOps architecture, MLflow can sit between model development and production deployment to improve experiment reproducibility, model versioning, and lifecycle management."

This is much stronger than:

> "MLflow is a tool used in MLOps."

---

## 22. Knowledge check

Try answering these **without looking back**.

### Beginner

**1.** What is an MLflow *run*?

**2.** What is the difference between a parameter and a metric?

**3.** Give three examples of artifacts.

### Intermediate

**4.** Why isn't MLflow itself a machine-learning training framework?

**5.** What's the difference between MLflow Tracking and Model Registry?

### Scenario

You trained 50 XGBoost models with different hyperparameters.

Your team wants to know:

> "Which model produced the highest F1 score, what hyperparameters did it use, and where is the trained model?"

**How would you use MLflow to solve this?**

---

## If You Remember Only 5 Things

1. **MLflow helps manage the ML lifecycle.**
2. A **run** represents an individual experiment/training execution.
3. **Parameters = what you chose; metrics = what you measured.**
4. **Artifacts = files produced by your experiment**, such as models and plots.
5. **Tracking records experiments; Model Registry manages model versions/lifecycle.**

---

## Learning Path

```text
MLflow Basics
     ↓
Experiment Tracking
     ↓
Artifacts & Model Logging
     ↓
Model Registry
     ↓
MLflow Projects / Packaging
     ↓
Model Serving
     ↓
Docker
     ↓
CI/CD
     ↓
Cloud Object Storage
     ↓
Kubernetes
     ↓
Production MLOps Architecture
```

The next practical step is to build a small **Scikit-learn + MLflow** project, then progressively turn it into a production-style MLOps pipeline.
