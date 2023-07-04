import os
import logging
import azure.functions as func
from azureml.core import Workspace, Experiment, Datastore
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(myblob: func.InputStream):
    # Set up Azure ML workspace and experiment
    logger.info("Setting up Azure ML workspace and experiment")

    # Authenticate with Azure
    tenant_id = "24b1c19c-1155-44af-bba1-549638587676"
    #tenant_id=os.environ.get("tenant_id")
    service_principal_id = os.environ.get("service_principal_id")
    service_principal_password = os.environ.get("service_principal_password")
    
    auth = ServicePrincipalAuthentication(tenant_id, service_principal_id,service_principal_password)

    workspace = Workspace.get(
        name=os.environ.get("workspace_name"),
        subscription_id=os.environ.get("subscription_id"),
        resource_group=os.environ.get("RESOURCE_GROUP_NAME"),
        auth=auth
        )
    experiment = Experiment(workspace=workspace, name="taskmlflow")

    logger.info("Registering Azure Blob datastores")

    blob_input_datastore_name = "inputdatastore"
    blob_output_datastore_name = "outputdatastore"

    Datastore.register_azure_blob_container(
        workspace=workspace,
        datastore_name=blob_input_datastore_name,
        account_name=os.environ.get("src_storage_acc_name"),
        container_name=os.environ.get("src_storage_container_name"),
        account_key=os.environ.get("src_account_key")
    )

    Datastore.register_azure_blob_container(
        workspace=workspace,
        datastore_name=blob_output_datastore_name,
        account_name=os.environ.get("dest_storage_acc_name"),
        container_name=os.environ.get("dest_storage_container_name"),
        account_key=os.environ.get("dest_account_key")
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

    script_name = "validate_and_combine.py"

    script_params = [
        "--input1", input_data_1,
        "--input2", input_data_2,
        "--output", output_data
    ]

    # Compute provisioning commented out for brevity

    logger.info("Creating validation and combination step")

    aml_compute = AmlCompute(workspace, 'worker-cluster')
    run_amlcompute = RunConfiguration()
    run_amlcompute.target = 'worker-cluster'
    run_amlcompute.environment.docker.enabled = True
    run_amlcompute.environment.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04'
    run_amlcompute.environment.python.user_managed_dependencies = False
    run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
        'azure-storage-blob',
        'joblib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'azureml-mlflow',
        'azure-ai-ml',
        'pyarrow',
        'ruamel.yaml',
        'matplotlib'
    ])

    validation_combination_step = PythonScriptStep(
        name="Validation and Combination",
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        script_name=script_name,
        arguments=script_params,
        inputs=[input_data_1, input_data_2],
        outputs=[output_data],
        compute_target=aml_compute,
        runconfig=run_amlcompute
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

    return "MLflow pipeline triggered successfully"
