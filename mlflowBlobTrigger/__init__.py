import os
import logging
import azure.functions as func
from azureml.core import Workspace, Experiment, Datastore
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_workspace():
    # Authenticate with Azure
    tenant_id = os.environ.get("tenant_id")
    service_principal_id = os.environ.get("service_principal_id")
    service_principal_password = os.environ.get("service_principal_password")

    auth = ServicePrincipalAuthentication(tenant_id,
                                          service_principal_id,
                                          service_principal_password)

    workspace = Workspace.get(
        name=os.environ.get("workspace_name"),
        subscription_id=os.environ.get("subscription_id"),
        resource_group=os.environ.get("RESOURCE_GROUP_NAME"),
        auth=auth
    )

    return workspace


def register_datastores(workspace):
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

    return blob_input_datastore_name, blob_output_datastore_name


def configure_pipeline_data(workspace, blob_output_datastore_name):
    logger.info("Configuring pipeline data")

    output_data = PipelineData(
        "output_data",
        datastore=Datastore(workspace, blob_output_datastore_name)
    )

    return output_data


def create_compute_target(workspace):
    compute_target_name = "mlflowcluster"
    min_nodes = 0
    max_nodes = 1
    vm_size = "STANDARD_DS3_V2"

    if compute_target_name not in workspace.compute_targets:
        logger.info("Creating Azure Machine Learning compute target")

        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            idle_seconds_before_scaledown=600
        )

        compute_target = ComputeTarget.create(workspace,
                                              compute_target_name,
                                              compute_config)
        compute_target.wait_for_completion(show_output=True)
    else:
        logger.info("Attaching to existing Azure Machine Learning compute target")
        compute_target = workspace.compute_targets[compute_target_name]

    return compute_target


def create_run_configuration(compute_target):
    run_amlcompute = RunConfiguration()
    run_amlcompute.target = compute_target.name
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

    return run_amlcompute


def create_validation_combination_step(output_data, input_data_1, input_data_2, compute_target, run_amlcompute):
    script_name = "validate_and_combine.py"
    script_params = [
        "--input1", input_data_1,
        "--input2", input_data_2,
        "--output", output_data
    ]

    validation_combination_step = PythonScriptStep(
        name="Validation and Combination",
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        script_name=script_name,
        arguments=script_params,
        inputs=[input_data_1, input_data_2],
        outputs=[output_data],
        compute_target=compute_target,
        runconfig=run_amlcompute
    )

    return validation_combination_step


def create_pipeline(workspace, experiment, validation_combination_step):
    logger.info("Creating pipeline")

    pipeline = Pipeline(workspace=workspace,
                        steps=[validation_combination_step])

    logger.info("Validating pipeline")

    pipeline.validate()

    return pipeline


def main(myblob: func.InputStream):
    # Set up Azure ML workspace and experiment
    logger.info("Setting up Azure ML workspace and experiment")

    workspace = setup_workspace()
    experiment = Experiment(workspace=workspace, name="taskmlflow")

    blob_input_datastore_name, blob_output_datastore_name = register_datastores(workspace)
    output_data = configure_pipeline_data(workspace, blob_output_datastore_name)

    compute_target = create_compute_target(workspace)
    run_amlcompute = create_run_configuration(compute_target)

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

    validation_combination_step = create_validation_combination_step(output_data, input_data_1, input_data_2,
                                                                    compute_target, run_amlcompute)

    pipeline = create_pipeline(workspace, experiment, validation_combination_step)

    logger.info("Submitting pipeline experiment")

    pipeline_run = experiment.submit(pipeline)

    logger.info("Pipeline is waiting for completion.")
    pipeline_run.wait_for_completion()

    return "MLflow pipeline triggered successfully"
