"""
For ETHZ users:
To use this script, you need to change for each computing node,
the project folder on each computing node and data folder.
"""


import subprocess
from abc import abstractmethod
import itertools


class Node(object):
    """
    Base class for describing computing nodes of experiments.
    For every new computing node, one should make a realization of this class.
    * :attr: 'address': str the address of the computing node
    * :attr: 'data_root': str the home folder of all data on the computing node
    """
    def __init__(self):
        self.address = None
        self.data_root = None
        self.project_root = None

    def get_address(self):
        return self.address

    def get_data_root(self):
        return self.data_root

    def get_project_root(self):
        return self.project_root

    @abstractmethod
    def run_experiment(self, experiment_arguments, gpus):
        """ Do the computation.
        :param experiment_arguments: string of arguments
        :return: [False/True] depending if job submission was successful
        """

class NoNode(Node):
    def run_experiment(self, experiment_arguments, gpus):
        command = 'python train.py' + experiment_arguments
        subprocess.Popen([command],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)


class Euler(Node):
    """ For executing computations on ETHZ Euler. """
    def __init__(self):
        super(Euler, self).__init__()
        self.address = "euler.ethz.ch"
        self.data_scratch = "$TMPDIR/data/"
        self.client = None
        self.project_path = "dental_imaging"

    def node_run_experiment(self, experiment_config, gpus, gpu_q, gpu_model, email, name, run_script):
        """ this is a non-blocking run """
        print("Sending a job to Euler.")
        N = '-N ' if email is True else ''
        if name != None:
            J = '-J ' + name + ' '
        else:
            J = ''

        # augment experiment arguments

        if experiment_config is not None:
            experiment_arguments = 'experiment=' +experiment_config

        else:
            experiment_arguments = ""

        command = 'source .bash_profile; cd ' + self.project_path + '; ' \
                  'module load gcc/8.2.0; ' \
                  'module load python_gpu/3.8.5; ' \
                  'module load cuda/11.3.1; ' \
                  'module load eth_proxy; ' \
                  'bsub ' + N + J + '-n '+str(int(gpus*2))+' -W ' +str(gpu_q)+ ':00 -R "rusage[mem=11000,ngpus_excl_p='+str(gpus)+']" ' \
                  "'python " + run_script + ' ' + experiment_arguments + "' "


        # "select[gpu_model0=='+gpu_model+']"
        
        # -R "select[gpu_mtotal0>=10240]" ' \
        # "bsub -n 20 -W 4:00 -R 'rusage[mem=4500, ngpus_excl_p=8]' -R 'select[gpu_model0==GeForceGTX1080Ti]' " \
            # "-W 1200 " \
        #TeslaV100_SXM2_32GB
        # add this if local scratch is necessary:
        # "'cp -r "+self.data_root+" "+self.data_scratch+"; " \
        # "bsub -R rusage[ngpus_excl_p=1,mem=8000,scratch=5000] "

        print(command)
        print("Sleep for 3 seconds..")
        import time
        time.sleep(3)

        # submit the job to Leonhard
        subprocess.Popen(["ssh", "%s" % "emanete@euler.ethz.ch", command],
                         shell=False,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

        print("Job sent!")

        return True


#def run_experiment(nodes, exp_name, models, repetitions, parameters):
def run_experiment(nodes, gpus, gpu_q,gpu_model, email, experiment,name,run_script):

        # find a computing node and run the experiment on it
        for node in nodes:
            print("Trying node: ", node.address)
            success = node.node_run_experiment(experiment, gpus, gpu_q, gpu_model, email, name, run_script)
            if success:
                print("Connection established!")
                break
            else:
                print("This node is busy.")
