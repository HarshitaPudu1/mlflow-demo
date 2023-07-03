import os
import logging
import azure.functions as func
import azureml.core
from azureml.core import Workspace, Experiment, Datastore
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import ServicePrincipalAuthentication
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(myblob: func.InputStream):
    # Set up Azure ML workspace and experiment
    logger.info("Setting up Azure ML workspace and experiment")


    # Authenticate with Azure
    tenant_id="24b1c19c-1155-44af-bba1-549638587676"
    service_principal_id = "970e2d10-b0dc-4821-9bc5-e0145f263e76"
    service_principal_password = "Sr48Q~bd7ZQKghY3~rGwXsiMIO7M2sTPeLczXa-1"
    
    auth = ServicePrincipalAuthentication(tenant_id, service_principal_id, service_principal_password)

    workspace = Workspace.get(
        name="demomlflowworkspace",
        subscription_id="3cfa681b-9a6f-4abf-9e24-e4f15f8da808",
        resource_group="demoAzure-Functions",
        auth=auth
        )
    experiment = Experiment(workspace=workspace, name="taskmlflow")

    logger.info("Registering Azure Blob datastores")

    blob_input_datastore_name = "inputdatastore2"
    blob_output_datastore_name = "outputdatastore2"

    Datastore.register_azure_blob_container(
        workspace=workspace,
        datastore_name=blob_input_datastore_name,
        account_name="demosrcblobaccstrg",
        container_name="demo-data",
        account_key=("zLszx1cX2mZthuP7P9qlJoBR2wsB344SC4qDCjk/"
                     "0ZtKGuRTojaUPkuFzYKWbhdvUNMH+s0Rvqug+AStNjnTvg==")
    )

    Datastore.register_azure_blob_container(
        workspace=workspace,
        datastore_name=blob_output_datastore_name,
        account_name="demodestaccstrg",
        container_name="demo-data",
        account_key=("D87qMS2dc1J2kON6ZXwBhAG38vV7yBr4cV5UfXcxkdzck2bcWgaK/"
                     "XC+F1H5hBWDx2s4YgJzpkCl+AStTDvfbg==")
    )

    logger.info("Configuring pipeline data")

    output_data = PipelineData(
        "output_data",
        datastore=Datastore(workspace, blob_output_datastore_name)
    )

    input_data_1 = DataReference(
        datastore=Datastore(workspace, blob_input_datastore_name),
        data_reference_name="departmentdata",
        path_on_datastore="/departmentsinput1.csv"
    )

    input_data_2 = DataReference(
        datastore=Datastore(workspace, blob_input_datastore_name),
        data_reference_name="employeedata",
        path_on_datastore="/employeesinput2.csv"
    )

    mlflow_env = azureml.core.Environment.from_conda_specification(
        name="mlflow-env",
        file_path=Path(__file__).parent / "conda.yml"
    )

    script_name = "validate_and_combine.py"

    script_params = [
        "--input1", input_data_1,
        "--input2", input_data_2,
        "--output", output_data
    ]

    compute_name = "taskmlflowinstance"

    # Compute provisioning commented out for brevity

    logger.info("Creating validation and combination step")


    # compute_config = ComputeInstance.provisioning_configuration(
    #     vm_size="Standard_DS2_v2"
    # )

    # try:
    #     compute_instance = ComputeTarget(workspace, compute_name)
    #     print("Found existing compute instance.")
    # except ComputeTargetException:
    #     compute_instance = ComputeInstance.create(workspace,
    #                                               compute_name,
    #                                               compute_config)
    #     compute_instance.wait_for_completion(show_output=True)

    #     # Handle Azure Function timeout
    #     raise TimeoutError("Compute instance creation timed out.")
    
    # while compute_instance.provisioning_state != 'Succeeded':
    #     print('Waiting for compute instance to be ready...')
    #     time.sleep(10)
    #     compute_instance = ComputeTarget(workspace, compute_name)
    validation_combination_step = PythonScriptStep(
        name="Validation and Combination",
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        script_name=script_name,
        arguments=script_params,
        inputs=[input_data_1, input_data_2],
        outputs=[output_data],
        compute_target=compute_name,
        runconfig={"environment": mlflow_env}
    )

    logger.info("Creating pipeline")

    pipeline = Pipeline(workspace=workspace,
                        steps=[validation_combination_step])

    logger.info("Validating pipeline")

    pipeline.validate()

    logger.info("Submitting pipeline experiment")

    pipeline_run = experiment.submit(pipeline)

    logger.info("Pipeline is waiting for completion.")
    pipeline_run.wait_for_completion()

    logger.info("MLflow pipeline triggered successfully")

