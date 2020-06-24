import os, sys, time
import numpy as np
import argparse
from utils.sweeper import Sweeper
from agents import *
from environments import *
from experiments import *
from utils.helpers import validate_output_folder

parser = argparse.ArgumentParser(description="Run an experiment based on parameters specified in a configuration file")
parser.add_argument('--config-file', default='config_files/prediction_avgcost-td.json',
                    help='location of the config file for the experiment (default: config_files/diff-q.json)')
parser.add_argument('--exp', default='run_exp_learning_prediction',
                    help='"run_exp_learning_prediction" or "run_exp_learning_control"')
parser.add_argument('--cfg-start', default=0)
parser.add_argument('--cfg-end', default=-1)
parser.add_argument('--output-folder', default='results/')

args = parser.parse_args()
output_folder = validate_output_folder(args.output_folder)
print(args.exp)
print(args.config_file)
sweeper = Sweeper(args.config_file)
cfg_start_idx = int(args.cfg_start)
cfg_end_idx = int(args.cfg_end) if args.cfg_end != -1 else sweeper.total_combinations
print(output_folder)
print('\n\nRunning configurations %d to %d...\n\n' % (cfg_start_idx, cfg_end_idx))

start_time = time.time()

for i in range(cfg_start_idx, cfg_end_idx):
    config = sweeper.get_one_config(i)
    print(config)
    env = getattr(sys.modules[__name__], config["env"])
    agent = getattr(sys.modules[__name__], config["agent"])
    experiment = getattr(sys.modules[__name__], args.exp)
    log = experiment(env, agent, config)
    log['params'] = config
    filename = "{}_{}".format(config['exp_name'], 15+i)
    print('Saving results in: %s\n**********\n' % (filename))
    np.save("{}{}".format(output_folder, filename), log)
    print("Time elapsed: {:.2} minutes\n\n".format((time.time() - start_time) / 60))
    os.system('sleep 0.5')

end_time = time.time()
print("Total time elapsed: {:.2} minutes".format((end_time - start_time) / 60))