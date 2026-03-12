from datetime import datetime, timedelta
import os

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from mlflow.tracking import MlflowClient


def ingest_data(**context):
    dataset_path = "/opt/airflow/data/Titanic-Dataset.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print("\n=== DATA INGESTION STARTED ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset shape: {df.shape}")

    missing_counts = df.isnull().sum()
    print("\nMissing values count per column:")
    print(missing_counts.to_string())

    context["ti"].xcom_push(key="dataset_path", value=dataset_path)
    print("\nXCom push successful: dataset_path stored.")
    print("=== DATA INGESTION COMPLETED ===\n")


def validate_data(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(task_ids="data_ingestion", key="dataset_path")

    if not dataset_path:
        raise AirflowFailException("dataset_path not found in XCom.")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print("\n=== DATA VALIDATION STARTED ===")

    total_rows = len(df)
    age_missing_pct = (df["Age"].isnull().sum() / total_rows) * 100
    embarked_missing_pct = (df["Embarked"].isnull().sum() / total_rows) * 100

    print(f"Age missing percentage: {age_missing_pct:.2f}%")
    print(f"Embarked missing percentage: {embarked_missing_pct:.2f}%")

    if age_missing_pct > 30:
        raise AirflowFailException(
            f"Validation failed: Age missing percentage is {age_missing_pct:.2f}%, greater than 30%."
        )

    if embarked_missing_pct > 30:
        raise AirflowFailException(
            f"Validation failed: Embarked missing percentage is {embarked_missing_pct:.2f}%, greater than 30%."
        )

    print("Validation passed.")
    print("=== DATA VALIDATION COMPLETED ===\n")


def handle_missing_values_func(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(task_ids="data_ingestion", key="dataset_path")

    if not dataset_path:
        raise AirflowFailException("dataset_path not found in XCom.")

    df = pd.read_csv(dataset_path)

    print("\n=== HANDLE MISSING VALUES STARTED ===")
    print(df[["Age", "Embarked"]].isnull().sum().to_string())

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    output_path = "/opt/airflow/data/processed_missing_values.csv"
    df.to_csv(output_path, index=False)

    ti.xcom_push(key="missing_values_path", value=output_path)
    print(f"Saved file at: {output_path}")
    print("=== HANDLE MISSING VALUES COMPLETED ===\n")


def feature_engineering_func(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(task_ids="data_ingestion", key="dataset_path")

    if not dataset_path:
        raise AirflowFailException("dataset_path not found in XCom.")

    df = pd.read_csv(dataset_path)

    print("\n=== FEATURE ENGINEERING STARTED ===")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    output_path = "/opt/airflow/data/feature_engineered.csv"
    df.to_csv(output_path, index=False)

    ti.xcom_push(key="feature_engineering_path", value=output_path)
    print(f"Saved file at: {output_path}")
    print("=== FEATURE ENGINEERING COMPLETED ===\n")


def merge_processed_data_func(**context):
    ti = context["ti"]

    missing_values_path = ti.xcom_pull(
        task_ids="handle_missing_values",
        key="missing_values_path",
    )
    feature_engineering_path = ti.xcom_pull(
        task_ids="feature_engineering",
        key="feature_engineering_path",
    )

    print("\n=== MERGE PROCESSED DATA STARTED ===")
    print(f"missing_values_path: {missing_values_path}")
    print(f"feature_engineering_path: {feature_engineering_path}")

    if not missing_values_path:
        raise AirflowFailException("missing_values_path not found in XCom.")
    if not feature_engineering_path:
        raise AirflowFailException("feature_engineering_path not found in XCom.")

    if not os.path.exists(missing_values_path):
        raise FileNotFoundError(f"Missing-values file not found: {missing_values_path}")
    if not os.path.exists(feature_engineering_path):
        raise FileNotFoundError(f"Feature-engineering file not found: {feature_engineering_path}")

    df_missing = pd.read_csv(missing_values_path)
    df_features = pd.read_csv(feature_engineering_path)

    df_missing["FamilySize"] = df_features["FamilySize"]
    df_missing["IsAlone"] = df_features["IsAlone"]

    merged_path = "/opt/airflow/data/final_processed_titanic.csv"
    df_missing.to_csv(merged_path, index=False)

    ti.xcom_push(key="processed_dataset_path", value=merged_path)

    print(f"Merged dataset saved to: {merged_path}")
    print(f"Merged dataset shape: {df_missing.shape}")
    print("=== MERGE PROCESSED DATA COMPLETED ===\n")


def data_encoding_func(**context):
    ti = context["ti"]
    processed_dataset_path = ti.xcom_pull(
        task_ids="merge_processed_data",
        key="processed_dataset_path",
    )

    if not processed_dataset_path:
        raise AirflowFailException("processed_dataset_path not found in XCom.")

    if not os.path.exists(processed_dataset_path):
        raise FileNotFoundError(f"Processed dataset not found: {processed_dataset_path}")

    df = pd.read_csv(processed_dataset_path)

    print("\n=== DATA ENCODING STARTED ===")
    print(f"Input dataset path: {processed_dataset_path}")
    print(f"Input dataset shape: {df.shape}")

    sex_mapping = {"male": 0, "female": 1}
    df["Sex"] = df["Sex"].map(sex_mapping)

    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    df["Embarked"] = df["Embarked"].map(embarked_mapping)

    columns_to_drop = ["Name", "Ticket", "Cabin"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns_to_drop)

    print("\nEncoded columns:")
    print("- Sex: male=0, female=1")
    print("- Embarked: S=0, C=1, Q=2")
    print(f"Dropped irrelevant columns: {existing_columns_to_drop}")
    print(f"Encoded dataset shape: {df.shape}")

    encoded_path = "/opt/airflow/data/encoded_titanic.csv"
    df.to_csv(encoded_path, index=False)

    ti.xcom_push(key="encoded_dataset_path", value=encoded_path)

    print(f"\nEncoded dataset saved to: {encoded_path}")
    print("=== DATA ENCODING COMPLETED ===\n")


def model_training_func(**context):
    """
    Task 6:
    - Start MLflow run
    - Log model type and hyperparameters
    - Train Logistic Regression
    - Log model artifact and dataset size
    """
    ti = context["ti"]
    encoded_dataset_path = ti.xcom_pull(
        task_ids="data_encoding",
        key="encoded_dataset_path",
    )

    if not encoded_dataset_path:
        raise AirflowFailException("encoded_dataset_path not found in XCom.")

    if not os.path.exists(encoded_dataset_path):
        raise FileNotFoundError(f"Encoded dataset not found: {encoded_dataset_path}")

    df = pd.read_csv(encoded_dataset_path)

    if "Survived" not in df.columns:
        raise AirflowFailException("Target column 'Survived' not found in dataset.")

    print("\n=== MODEL TRAINING STARTED ===")
    print(f"Encoded dataset path: {encoded_dataset_path}")
    print(f"Dataset shape used for training: {df.shape}")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    dag_run_conf = context.get("dag_run").conf if context.get("dag_run") else {}

    test_size = float(dag_run_conf.get("test_size", 0.2))
    random_state = int(dag_run_conf.get("random_state", 48))
    max_iter = int(dag_run_conf.get("max_iter", 5000))
    model_name = dag_run_conf.get("model_type", "RandomForest")
    experiment_name = "Titanic_Survival_Prediction_Final_1"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Train-test split completed.")

    # Save test data before MLflow run
    X_test_path = "/opt/airflow/data/X_test.csv"
    y_test_path = "/opt/airflow/data/y_test.csv"

    X_test.to_csv(X_test_path, index=False)
    y_test.to_frame(name="Survived").to_csv(y_test_path, index=False)

    print(f"Saved X_test to: {X_test_path}")
    print(f"Saved y_test to: {y_test_path}")

    model_name == "RandomForest"
    from sklearn.ensemble import RandomForestClassifier

    n_estimators = int(dag_run_conf.get("n_estimators", 100))
    max_depth = dag_run_conf.get("max_depth", None)
    if max_depth is not None:
        max_depth = int(max_depth)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    print(f"Using RandomForest with n_estimators={n_estimators}, max_depth={max_depth}")

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_name)
    print(f"Using MLflow experiment: {experiment_name}")

    run_id = None

    try:
        with mlflow.start_run(run_name="logistic_regression_run") as run:
            run_id = run.info.run_id
            print(f"MLflow run started with run_id: {run_id}")

            model.fit(X_train, y_train)
            print("Model training completed.")

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            if model_name == "LogisticRegression":
                mlflow.log_param("max_iter", max_iter)
            elif model_name == "RandomForest":
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)

            mlflow.log_param("train_rows", int(len(X_train)))
            mlflow.log_param("test_rows", int(len(X_test)))
            mlflow.log_param("total_rows", int(len(df)))
            mlflow.log_param("num_features", int(X.shape[1]))

            print("Parameters logged to MLflow.")

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

            print("Model artifact logged to MLflow.")

        # Push XCom only after MLflow run closes successfully
        ti.xcom_push(key="mlflow_run_id", value=run_id)
        ti.xcom_push(key="X_test_path", value=X_test_path)
        ti.xcom_push(key="y_test_path", value=y_test_path)

        print(f"XCom push successful. run_id: {run_id}")
        print("=== MODEL TRAINING COMPLETED ===\n")

        return run_id

    except Exception as e:
        print(f"MODEL TRAINING ERROR: {str(e)}")
        raise AirflowFailException(f"Model training failed: {str(e)}")
    
def model_evaluation_func(**context):
    """
    Task 7:
    - Compute Accuracy, Precision, Recall, F1-score
    - Log all metrics to MLflow
    - Push accuracy using XCom
    """
    ti = context["ti"]

    mlflow_run_id = ti.xcom_pull(task_ids="model_training", key="mlflow_run_id")
    x_test_path = ti.xcom_pull(task_ids="model_training", key="X_test_path")
    y_test_path = ti.xcom_pull(task_ids="model_training", key="y_test_path")

    if not mlflow_run_id:
        raise AirflowFailException("mlflow_run_id not found in XCom.")
    if not x_test_path:
        raise AirflowFailException("X_test_path not found in XCom.")
    if not y_test_path:
        raise AirflowFailException("y_test_path not found in XCom.")

    if not os.path.exists(x_test_path):
        raise FileNotFoundError(f"X_test file not found: {x_test_path}")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"y_test file not found: {y_test_path}")

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["Survived"]

    model_uri = f"runs:/{mlflow_run_id}/model"

    print("\n=== MODEL EVALUATION STARTED ===")
    print(f"MLflow run_id: {mlflow_run_id}")
    print(f"Model URI: {model_uri}")

    mlflow.set_tracking_uri("http://mlflow:5000")
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    ti.xcom_push(key="accuracy", value=float(accuracy))

    print("Metrics logged to MLflow successfully.")
    print("Accuracy pushed to XCom successfully.")
    print("=== MODEL EVALUATION COMPLETED ===\n")

def choose_model_path(**context):
    """
    Task 8:
    - Read accuracy from XCom
    - If accuracy >= 0.80 -> register_model
    - Else -> reject_model
    """
    ti = context["ti"]
    accuracy = ti.xcom_pull(task_ids="model_evaluation", key="accuracy")

    if accuracy is None:
        raise AirflowFailException("Accuracy not found in XCom from model_evaluation task.")

    accuracy = float(accuracy)

    print("\n=== BRANCHING LOGIC STARTED ===")
    print(f"Model accuracy received from XCom: {accuracy:.4f}")

    if accuracy >= 0.80:
        print("Accuracy is >= 0.80. Proceeding to register_model.")
        return "register_model"
    else:
        print("Accuracy is < 0.80. Proceeding to reject_model.")
        return "reject_model"

def choose_model_path(**context):
    """
    Task 8:
    - Read accuracy from XCom
    - If accuracy >= 0.80 -> register_model
    - Else -> reject_model
    """
    ti = context["ti"]
    accuracy = ti.xcom_pull(task_ids="model_evaluation", key="accuracy")

    if accuracy is None:
        raise AirflowFailException("Accuracy not found in XCom from model_evaluation task.")

    accuracy = float(accuracy)

    print("\n=== BRANCHING LOGIC STARTED ===")
    print(f"Model accuracy received from XCom: {accuracy:.4f}")

    if accuracy >= 0.80:
        print("Accuracy is >= 0.80. Proceeding to register_model.")
        return "register_model"
    else:
        print("Accuracy is < 0.80. Proceeding to reject_model.")
        return "reject_model"

def register_model_func(**context):
    """
    Task 9:
    - Register approved model in MLflow Model Registry
    """
    ti = context["ti"]

    accuracy = ti.xcom_pull(task_ids="model_evaluation", key="accuracy")
    run_id = ti.xcom_pull(task_ids="model_training", key="mlflow_run_id")

    if accuracy is None:
        raise AirflowFailException("Accuracy not found in XCom.")

    if run_id is None:
        raise AirflowFailException("MLflow run_id not found in XCom.")

    accuracy = float(accuracy)

    model_name = "TitanicSurvivalModel"
    model_uri = f"runs:/{run_id}/model"

    print("\n=== MODEL REGISTRATION STARTED ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"Registered model name: {model_name}")
    print(f"Registered model version: {registered_model.version}")

    # Optional: add tags/description for clarity
    client.set_model_version_tag(
        name=model_name,
        version=registered_model.version,
        key="accuracy",
        value=str(round(accuracy, 4)),
    )

    client.set_model_version_tag(
        name=model_name,
        version=registered_model.version,
        key="status",
        value="approved",
    )

    ti.xcom_push(key="registered_model_name", value=model_name)
    ti.xcom_push(key="registered_model_version", value=registered_model.version)

    print("=== MODEL REGISTRATION COMPLETED ===\n")


def reject_model_func(**context):
    """
    Task 9:
    - Log rejection reason if accuracy is low
    """
    ti = context["ti"]

    accuracy = ti.xcom_pull(task_ids="model_evaluation", key="accuracy")
    run_id = ti.xcom_pull(task_ids="model_training", key="mlflow_run_id")

    if accuracy is None:
        raise AirflowFailException("Accuracy not found in XCom.")

    accuracy = float(accuracy)

    rejection_reason = (
        f"Model rejected because accuracy {accuracy:.4f} is below threshold 0.80"
    )

    print("\n=== MODEL REJECTION STARTED ===")
    print(rejection_reason)

    mlflow.set_tracking_uri("http://mlflow:5000")

    if run_id:
        client = MlflowClient()
        client.log_batch(
            run_id=run_id,
            params=[],
            metrics=[],
            tags=[
                {"key": "model_status", "value": "rejected"},
                {"key": "rejection_reason", "value": rejection_reason},
            ],
        )

    ti.xcom_push(key="rejection_reason", value=rejection_reason)

    print("=== MODEL REJECTION COMPLETED ===\n")

default_args = {
    "owner": "student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="mlops_airflow_mlflow_pipeline",
    default_args=default_args,
    description="End-to-end Titanic ML pipeline with Airflow and MLflow",
    start_date=datetime(2026, 3, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "titanic", "airflow", "mlflow"],
) as dag:

    start = EmptyOperator(task_id="start")

    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data,
    )

    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=validate_data,
        retries=1,
        retry_delay=timedelta(minutes=1),
    )

    handle_missing_values = PythonOperator(
        task_id="handle_missing_values",
        python_callable=handle_missing_values_func,
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_func,
    )

    merge_processed_data = PythonOperator(
        task_id="merge_processed_data",
        python_callable=merge_processed_data_func,
    )

    data_encoding = PythonOperator(
        task_id="data_encoding",
        python_callable=data_encoding_func,
    )

    model_training = PythonOperator(
        task_id="model_training",
        python_callable=model_training_func,
    )

    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation_func,
    )

    branch_on_accuracy = BranchPythonOperator(
        task_id="branch_on_accuracy",
        python_callable=choose_model_path,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_func,
    )

    reject_model = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model_func,
    )


    end = EmptyOperator(
    task_id="end",
    trigger_rule="none_failed_min_one_success",
    )

    start >> data_ingestion >> data_validation
    data_validation >> [handle_missing_values, feature_engineering]
    [handle_missing_values, feature_engineering] >> merge_processed_data
    merge_processed_data >> data_encoding >> model_training >> model_evaluation
    model_evaluation >> branch_on_accuracy
    branch_on_accuracy >> register_model >> end
    branch_on_accuracy >> reject_model >> end