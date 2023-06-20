from azureml.core import Workspace, Dataset, Environment

# Load the stored workspace
ws = Workspace.from_config()

# Get the registered dataset from azure
dataset = Dataset.get_by_name(ws, name='input_images')

## Try with our saved image
env = Environment.get(workspace=ws, name="waste-env-gpu")

# get our compute target
compute_target = ws.compute_targets["gpu-cluster-NC6"]


from azureml.core import Experiment

# define the expiriment
exp = Experiment(workspace=ws, name='image_segmentation_third_try')

from azureml.core import ScriptRunConfig

# setup the run details
src = ScriptRunConfig(source_directory='C:/Users/DavidS10/PycharmProjects/pythonProject/image_classification/code',
                      script='train.py',
                      arguments=['--data-path', dataset.as_mount()],
                      compute_target=compute_target,
                      environment=env)

# Submit the model to azure!
run = exp.submit(config=src)