import logging
from datetime import datetime
import azure.functions as func
import azureml
from azureml.core import Workspace, Experiment, Datastore
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep
import pandas as pd
from azureml.core.compute import ComputeTarget, ComputeInstance
from azureml.core.compute_target import ComputeTargetException
import os
from pathlib import Path


def main(myblob: func.InputStream):
    # Set up Azure ML workspace and experiment
    workspace = Workspace.get(name="demomlworkspace", subscription_id="3cfa681b-9a6f-4abf-9e24-e4f15f8da808", resource_group="demoAzure-Functions")
    experiment = Experiment(workspace=workspace, name="taskmlflow")
    blob_output_datastore_name = blob_input_datastore
    blob_input_datastore_name = blob_output_datastore
    blob_input_datastore = Datastore.register_azure_blob_container(
           workspace=workspace,
           datastore_name=blob_input_datastore_name,
           account_name="demosrcblobaccstrg",
           container_name="demo-data",
           account_key="zLszx1cX2mZthuP7P9qlJoBR2wsB344SC4qDCjk/0ZtKGuRTojaUPkuFzYKWbhdvUNMH+s0Rvqug+AStNjnTvg==")
    blob_output_datastore = Datastore.register_azure_blob_container(
           workspace=workspace,
           datastore_name=blob_output_datastore_name,
           account_name="demodestaccstrg",
           container_name="demo-data",
           account_key="D87qMS2dc1J2kON6ZXwBhAG38vV7yBr4cV5UfXcxkdzck2bcWgaK/XC+F1H5hBWDx2s4YgJzpkCl+AStTDvfbg==")
    output_data = PipelineData("output-data", datastore=Datastore(workspace, blob_output_datastore_name))
    input_data_1 = DataReference(datastore=Datastore(workspace, blob_input_datastore_name),data_reference_name="departmentdata", 
                                        path_on_datastore="/departmentsinput1.csv")
    input_data_2 = DataReference(datastore=Datastore(workspace, blob_input_datastore_name),data_reference_name="employeedata", 
                                        path_on_datastore="/employeesinput2.csv")
    input_data_version = datetime.now().strftime("%Y%m%d%H%M%S")
    mlflow_env = azureml.core.Environment.from_conda_specification(name="mlflow-env", file_path=Path(__name__).parent / "conda.yml")
    script_name = "validate_and_combine.py"
    script_params = [
        "--input1", input_data_1,
        "--input2", input_data_2,
        "--output", output_data,
    ]
    compute_name = "taskmlflow-instance"
    compute_config = ComputeInstance.provisioning_configuration(
        vm_size="Standard_DS2_v2"
    )
    try:
        compute_instance = ComputeTarget(workspace, compute_name)
        print("Found existing compute instance.")
    except ComputeTargetException:
        compute_instance = ComputeInstance.create(workspace, compute_name, compute_config)
        compute_instance.wait_for_completion(show_output=True)
    validation_combination_step = PythonScriptStep(
        name="Validation and Combination",
        source_directory = os.path.dirname(os.path.realpath(__file__)),
        script_name=script_name,
        arguments=script_params,
        inputs=[input_data_1, input_data_2],
        outputs=[output_data],
        compute_target=compute_name,
        runconfig={
            "environment": mlflow_env
        }
    )
    pipeline = Pipeline(workspace=workspace, steps=[validation_combination_step])
    print(pipeline)
    pipeline.validate()
    pipeline_run = experiment.submit(pipeline)
    print("pipeline Waiting")
    pipeline_run.wait_for_completion()
    logging.info("MLflow pipeline triggered successfully")
