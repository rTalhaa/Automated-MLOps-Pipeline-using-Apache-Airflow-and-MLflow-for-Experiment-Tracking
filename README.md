# Automated-MLOps-Pipeline-using-Apache-Airflow-and-MLflow-for-Experiment-Tracking
This project implements an end-to-end MLOps pipeline using Apache Airflow and MLflow on the Titanic dataset. Airflow orchestrates tasks (data ingestion, validation, preprocessing, training, evaluation, and decision branches), while MLflow tracks experiments (logging parameters, metrics, and model artifacts) and hosts the model registry.

# MLOps Pipeline – Airflow + MLflow

This repository implements a simple **MLOps pipeline** using **Apache Airflow for orchestration** and **MLflow for experiment tracking and model management**.  
The pipeline trains and evaluates machine learning models on the **Titanic dataset** and demonstrates automated data processing, experiment tracking, model evaluation, and conditional model registration.

---

# Architecture Explanation (Airflow + MLflow Interaction)

## What components are used in the architecture?

- **Apache Airflow** – orchestrates the ML workflow using a DAG.
- **MLflow** – tracks experiments, logs parameters/metrics, and stores trained models.
- **PostgreSQL** – stores Airflow metadata.
- **Docker** – containerizes all services for reproducibility.

## How do Airflow and MLflow interact?

1. Airflow runs the ML pipeline tasks defined in the DAG.
2. During the **model training task**, an MLflow experiment run is started.
3. MLflow logs:
   - model type
   - hyperparameters
   - dataset size
   - trained model artifact.
4. The **evaluation task logs accuracy metrics** to MLflow.
5. Airflow uses **branching logic** to decide whether the model should be **registered or rejected**.

---

# DAG Structure and Dependency Explanation

## What tasks exist in the DAG?

1. **Data Ingestion**
   - Load Titanic dataset
   - Print dataset shape
   - Push dataset path via XCom

2. **Data Validation**
   - Check missing values in `Age` and `Embarked`
   - Raise exception if missing percentage exceeds threshold

3. **Parallel Processing**
   - Handle missing values
   - Perform feature engineering (`FamilySize`, `IsAlone`)

4. **Data Encoding**
   - Encode categorical variables (`Sex`, `Embarked`)
   - Drop irrelevant columns

5. **Model Training**
   - Start MLflow run
   - Train Logistic Regression or Random Forest
   - Log parameters and dataset size
   - Log model artifact

6. **Model Evaluation**
   - Calculate model accuracy
   - Log evaluation metrics

7. **Branching Logic**
   - If **accuracy ≥ 0.80** → Register model
   - If **accuracy < 0.80** → Reject model

8. **Model Registration**
   - Register approved model in MLflow Model Registry

9. **Model Rejection**
   - Log rejection reason

## Why are there no cyclic dependencies?

Airflow pipelines are **Directed Acyclic Graphs (DAGs)**, meaning tasks only move forward in one direction.  
No task depends on a later task, ensuring there are **no loops or cyclic dependencies**.

---

# Experiment Comparison Analysis

## How were experiments conducted?

The pipeline was executed **three times with different hyperparameters** to compare performance.

| Run | Model | Hyperparameters |
|----|----|----|
| Run 1 | Logistic Regression | max_iter = 1000 |
| Run 2 | Logistic Regression | max_iter = 3000 |
| Run 3 | Random Forest | n_estimators = 200, max_depth = 6 |

## How was the best model identified?

MLflow experiment tracking was used to compare runs based on **accuracy and logged metrics**.  
Using the **MLflow UI comparison view**, the model with the **highest accuracy** was selected as the best-performing model.

---

# Failure and Retry Explanation

## How does the pipeline handle failures?

Airflow provides built-in retry and failure management.

### Example failure scenarios

**1. Data validation failure**
- If missing values exceed the threshold, an exception is raised.
- Airflow marks the task as failed and retries according to DAG settings.

**2. Infrastructure or service errors**
- Issues such as MLflow connection problems or container configuration errors are logged.

Airflow allows debugging using:
- **Task logs**
- **Graph View**
- **Retry mechanisms**

---

# Reflection Questions on Production Deployment

## What improvements would be required for production?

**1. Distributed execution**
- Use **CeleryExecutor or KubernetesExecutor** for scalable pipelines.

**2. Data versioning**
- Track dataset versions to ensure reproducible experiments.

**3. Model monitoring**
- Monitor deployed models to detect performance drift.

**4. CI/CD integration**
- Automatically retrain models when new data or code changes occur.

**5. Security and access control**
- Restrict access to MLflow experiments and model registry.

---

# GitHub Repository

## What files are included in the repository?

```
mlops-airflow-mlflow-pipeline
│
├── dags/
│   └── mlops_airflow_mlflow_pipeline.py
│
├── data/
│   └── titanic.csv
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# How to Run the Project

## 1. Clone the repository

```
git clone <repository-link>
cd mlops-airflow-mlflow-pipeline
```

## 2. Start all services

```
docker compose up --build
```

This starts:
- Airflow Webserver
- Airflow Scheduler
- PostgreSQL
- MLflow Server

---

## 3. Access Airflow UI

```
http://localhost:8080
```

Default credentials:

```
Username: airflow
Password: airflow
```

---

## 4. Access MLflow UI

```
http://localhost:5000
```

This interface shows:
- experiment runs
- logged parameters
- metrics
- artifacts
- registered models

---

## 5. Run the pipeline

1. Open Airflow UI  
2. Enable DAG **mlops_airflow_mlflow_pipeline**  
3. Trigger the DAG  
4. Monitor execution in **Graph View**

---

# Technologies Used

- Apache Airflow
- MLflow
- Docker
- Python
- Scikit-learn
- Pandas

---

# Key Learning Outcomes

- Building automated ML pipelines with Airflow
- Tracking ML experiments using MLflow
- Logging models, metrics, and parameters
- Using branching logic for automated model approval

- Running reproducible pipelines using Docker
