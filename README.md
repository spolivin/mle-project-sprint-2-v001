# Improving baseline model

## Business task

The project follows the process started in the previous project. We have developed a basic solution for the flat price prediction task in the form of an ML-model as well as created data preprocessing pipeline via Airflow. 

Neverhtless, the objective now is to come up with an alternative solution which could improve the metrics and thus optimize our approach starting with the base solution and the built data pipelines. In other words, we need to make the process reproducible and improve on the previously used metrics.

## ML task

In order to improve the key model metrics one needs to make use of feature engineering, feature selection and hyperparameters optimization techniques. 

## MLflow server launching

In order to launch Mlflow server one needs to execute the following commands from the root directory:

```bash
cd mlflow_server
sh run_mlflow_server.sh
```

After the server has been successfilly launched, we can launch the first logging of the baseline model and its artifacts in the following way:

```bash
python log_baseline_model.py
```
> Before running the command, one needs to temporarily change the version of the `scikit-learn` library to `1.4.1.post1`, since the binary version of the model in `pkl` format has been created in the other version of the library
