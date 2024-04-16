import argparse
import pickle
import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', required=True, help='result pkl file path')
args = parser.parse_args()

path_res = args.file

with open(path_res, 'rb') as f:
    res_optuna = pickle.load(f)

# TODO: How do I decide on this weighting between cost and failure? Just 1.0?
all_res = [trial.values[0] + 1.0 * (1.0 - trial.values[1]) for trial in res_optuna]
idx = np.argsort(all_res)

res_optuna = np.array(res_optuna)

for trial in res_optuna[idx][:10]:
    print("Cost, Validity:", trial.values)

print("Show Best params.")
id_s = 0
while id_s != -1:
    id_s = input("Select an index. Give -1 to exit: ")
    if id_s == '-1':
        break
    else:
        try:
            id_s = int(id_s)
            print(res_optuna[idx][id_s].params)
        except:
            print("exit")
            exit(0)

