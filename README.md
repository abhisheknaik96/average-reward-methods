# average-reward-methods

Accompanying code for the paper "Learning and Planning in Average-Reward Markov Decision Processes" by Yi Wan*, Abhishek Naik*, Rich Sutton.

---

- `agents/` folder contains all the algorithms.
- `environments/` folder contains all the environments.
- `config_files/` folder contains sample configuration files for various experiments.
- `experiments.py` contains methods to run different kinds of experiments, e.g., prediction, control.
- `run_exp.py` runs an experiment based on command-line arguments outlined below.

---

A typical experiment looks like:
```bash
python run_exp.py --exp run_exp_learning_control_no_eval --config-file config_files/control_AccessControl_diff-q.json --output-folder results/control/AccessControl 
```
where,
- `exp`: the experiment to be run. For prediction and control, this will generally be `run_exp_learning_prediction` or `run_exp_learning_control_no_eval`. Check `experiments.py` for full documentation and use-cases.
- `config-file`: the file with all the experiment configurations
- `output-folder`: the location where all the result-logs will be stored

Optional parameters for deploying experiments at scale:
- `cfg-start`: the start index of the list of configurations for this script
- `cfg-end`: the end index of the list of configurations for this script (refer to `utils/sweeper.py` for more details) 

Check out the jupyter notebook `learning_planning_exps.ipynb` for sample experiments and the plots reported in the paper.