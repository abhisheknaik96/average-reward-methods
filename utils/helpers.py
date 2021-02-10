import glob
import numpy as np
import os

def validate_output_folder(path):
    """checks if folder exists. If not, creates it and returns its name"""
    if path[-1] != '/':
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def all_files_with_prefix_and_suffix(location, prefix, suffix):
    """returns a list of all files in the 'location' starting with the given prefix"""
    if location[-1] != '/':
        location += '/'
    files = glob.glob(location + prefix + '*' + suffix)

    return files


def get_weights_from_npy(filename):
    data = np.load(filename, allow_pickle=True).item()
    weights = np.mean(data['weights_final'], axis=0)
    print(weights)

    return weights


if __name__ == "__main__":
    print(all_files_with_prefix_and_suffix("../results/",
                                           "RiverSwim_DifferentialQlearningAgent_sensitivity_fine_",
                                           ".npy"))
