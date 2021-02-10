import os
import numpy as np
from tqdm import tqdm
import json
import glob

import matplotlib.pyplot as plt
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams["font.family"] = "Palatino"
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'figure.figsize': [8,5]})

from helpers import all_files_with_prefix_and_suffix


def parse_data_ac(filename_prefix, item, location='../results/', param1='alpha_w', param2='dummy',
                  title=None, firstn=-1, type='heatmap', num_sa=None, logbase=-1, ylabel='RMSVE\n(TVR)'):

    files = all_files_with_prefix_and_suffix(location, filename_prefix, '*.npy')
    data_all = {}
    assert len(files) > 0, 'No files found with the prefix: ' + location + filename_prefix
    print('Filename\t%s\t%s\tValue\n' % (param1, param2))
    for file in files:
        data = np.load(file, allow_pickle=True).item()
        param1_value = data['params']['agent_parameters'][param1]
        if param1_value not in data_all:
            data_all[param1_value] = {}
        param2_value = data['params']['agent_parameters'][param2] if param2!='dummy' else -1
        if param2_value not in data_all[param1_value]:
            data_all[param1_value][param2_value] = {}
        data_to_eval = data[item][:,:firstn]
        mean = np.mean(data_to_eval)
        stderr = np.std(data_to_eval) / np.sqrt(data_to_eval.size)
        data_all[param1_value][param2_value] = (mean, stderr)
        print('%s\t%s\t%s\t%.3f\t%.3f' % (file[-10:-3], param1_value, param2_value, mean, stderr))

    X = sorted(data_all)
    Y = sorted(data_all[np.random.choice(list(data_all.keys()))])
    Z_mean = np.zeros((len(X), len(Y)))
    Z_stderr = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            try:
                Z_mean[i][j] = data_all[x][y][0]
                Z_stderr[i][j] = data_all[x][y][1]
            except:
                Z_mean[i][j] = np.nan
                Z_stderr[i][j] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(5.5,4))

    for i, param_value in enumerate(X):
        ax.errorbar(Y, Z_mean[i], yerr=Z_stderr[i], label=param1+'='+str(param_value))#, color='color')
    ax.grid(b=True, axis='y', alpha=0.5, linestyle='--')
    #     ax.legend()
    #     plt.xlabel(param2)
    #     plt.ylabel('RMSVE\n(TVR)', rotation=0, labelpad=40)
    #     ax.set_title(title)
    plt.ylim(2.05, 2.6)
    if logbase != -1:
        plt.xscale('log', basex=logbase)
    ax.set_xticks(Y)
    ax.set_xticklabels(Y)

    fig.tight_layout()
    assert os.path.isfile(location + title + '.png') == False, "File already exists. Don't overwrite!"
    plt.savefig(location + title + '.png')#, dpi=1200)
    # plt.show()


if __name__ == '__main__':

    parse_data_ac('AC_DiffQ_eps_0.1_eta_0.01', 'rewards_all_train', param1='eta', param2='alpha_w',
                  title='AC_DiffQ_eps_0.1_eta_0.01_sensitivity', location='./results/rebuttal/AC/',
                  firstn=-1, type='u-curves', logbase=10, ylabel='Average\nreward\nover\ntraining')