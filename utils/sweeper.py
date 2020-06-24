# Based on the sweeper.py file in
# https://github.com/muhammadzaheer/classic-control/blob/0f075ee2951de01d063bc1d069b28bf25167af20/sweeper.py

import json


class Sweeper:
    """
    The purpose of this class is to take an index, identify a configuration
    of hyper-parameters and create a Config object

    Important: parameters part of the sweep are provided in a list
    """
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config_dict = json.load(f)
        self.total_combinations = 1

        if 'sweep_parameters' in self.config_dict['agent_parameters']:
            sweep_params = self.config_dict['agent_parameters']['sweep_parameters']
            # calculating total_combinations
            tc = 1
            for params, values in sweep_params.items():
                tc = tc * len(values)
            self.total_combinations = tc

    def get_one_config(self, idx):
        """replaces the range of values by a single value based on the index idx"""
        cfg = {x: self.config_dict[x] for x in self.config_dict.keys() if x not in ['agent_parameters']}
        cfg['agent_parameters'] = {}
        fixed_params = self.config_dict['agent_parameters']['fixed_parameters']
        for param, value in fixed_params.items():
            cfg['agent_parameters'][param] = value
        if 'sweep_parameters' in self.config_dict['agent_parameters']:
            sweep_params = self.config_dict['agent_parameters']['sweep_parameters']
            cumulative = 1
            for param, values in sweep_params.items():
                cfg['agent_parameters'][param] = values[int(idx/cumulative) % len(values)]
                cumulative *= len(values)
        return cfg


if __name__ == '__main__':
    sweeper = Sweeper("../config_files/diff-q.json")
    print(sweeper.total_combinations)
    for i in range(sweeper.total_combinations):
        print(sweeper.get_one_config(i))
