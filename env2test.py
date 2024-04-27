from environmentv2 import Environment
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.environments import utils

BATCH_SIZE = 4


def create_environment():
    return Environment()


def main(argv=None):  # Accept argv even if you don't use it
    if argv is None:
        argv = []
    parallel_environment = ParallelPyEnvironment([create_environment] * BATCH_SIZE)
    
    try:
        utils.validate_py_environment(parallel_environment, episodes=5)
        print("Environment validation succeeded.")
    except Exception as e:
        print("Environment validation failed:", e)

    parallel_environment.close()

    
if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
