import joblib
import json
import os

import mlflow
import pandas as pd

EXPERIMENT_NAME = "PROJECT_SPRINT_2"
RUN_NAME = "baseline_model_logging"
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
REGISTRY_MODEL_NAME = "baseline_model"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")


def log_base_model() -> None:
    """Logs the baseline model using Mlflow."""

    # Loading CV-results
    with open("cv_results/cv_res_flats.json") as json_file:
        metrics = json.load(json_file)

    # Loading the trained model
    with open("models/fitted_model_flats.pkl", "rb") as fd:
        model = joblib.load(fd)

    # Retrieving Catboost model parameters
    model_params = model["catboostregressor"].get_params()

    # Reading the data on which the model was trained
    data = pd.read_csv("data/initial_data_flats.csv")

    # Creating/using an existing Mlflow-experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    # Setting model details
    pip_requirements = "./requirements_baseline.txt"
    features = data.drop("price", axis=1)
    prediction = model.predict(features)
    signature = mlflow.models.infer_signature(features, prediction)
    input_example = features[:10]

    # Initiating a logging procedure
    with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
        # Retrieving run identifier
        run_id = run.info.run_id

        # Logging training data
        mlflow.log_artifact("data/initial_data_flats.csv", "dataframe")

        # Logging metrics from CV-results
        mlflow.log_metrics(metrics)

        # Logging model parameters
        mlflow.log_params(model_params)

        # Registering the model in Mlflow Registry
        model_info = mlflow.sklearn.log_model(
            registered_model_name=REGISTRY_MODEL_NAME,
            sk_model=model,
            pip_requirements=pip_requirements,
            signature=signature,
            input_example=input_example,
            await_registration_for=60,
            artifact_path="models",
        )


if __name__ == "__main__":
    log_base_model()
