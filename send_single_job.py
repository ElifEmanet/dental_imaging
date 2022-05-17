"""
Follows Pytorch Lightning Hydra project structure
Assumes the existence of computation module inside src.utils

# Run from root folder with: python scripts/send_single_job.py

"""

from src.utils import computation


nodes = [computation.Euler()]
gpus = 1
gpu_q = 4 # define job time queue (default: 4h)
email = False # if True an email is sent when the job is concluded
experiment = 'example' # choose experiment config
name = 'test'  # name of job (only for cluster)
gpu_model = 'GeForceRTX2080Ti'#'TeslaV100_SXM2_32GB' (https://scicomp.ethz.ch/wiki/Using_the_batch_system#GPU)
run_script = 'train.py' #train.py test.py


computation.run_experiment(nodes,
                           gpus,
                           gpu_q,
                           gpu_model,
                           email,
                           experiment,
                           name,
                           run_script)