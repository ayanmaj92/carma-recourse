import pickle
import argparse
import numpy as np


if __name__ == '__main__':
    FLAG = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', '-res', type=str, help='Path to per-datum results pkl file.')
    args = parser.parse_args()
    with open(args.res_path, 'rb') as f:
        per_dat = pickle.load(f)
    act_d = {}
    lens_d = {}
    if 'oracle' not in args.res_path:
        for k in per_dat.keys():
            acts = per_dat[k]
            lens_list = [len(act.keys()) for act in acts]
            un, cnts = np.unique(lens_list, return_counts=True)
            dct_len = dict(zip(un, cnts))
            for j in dct_len:
                if j not in lens_d:
                    lens_d[j] = [dct_len[j]]
                else:
                    lens_d[j].append(dct_len[j])
            act_feat_idxs = [a for act in acts for a in list(act.keys())]
            uniq, counts = np.unique(act_feat_idxs, return_counts=True)
            if FLAG:
                print(k, uniq, counts)
                counts = (counts / len(acts)) * 100
            dct = dict(zip(uniq, counts))
            for i in dct:
                if i in act_d:
                    act_d[i].append(dct[i])
                else:
                    act_d[i] = [dct[i]]
        max_len = max([len(act_d[i]) for i in act_d])
        for i in act_d:
            for _ in range(max_len - len(act_d)):
                act_d[i].append(0.0)

        TOT = len(per_dat.keys()) * len(acts)
        print("Means")
        if FLAG:
            means = {k: np.mean(v) for k, v in act_d.items()}
        else:
            means = {k: np.sum(v) / TOT for k, v in act_d.items()}
        print(means)
        TOT = len(per_dat.keys()) * len(acts)
        print("AVERAGE NUMBER OF FEATS INTERVENED")
        print({k: np.sum(v) / TOT for k, v in lens_d.items()})
    else:
        lens_list = [len(act.keys()) for act in per_dat]
        un, cnts = np.unique(lens_list, return_counts=True)
        cnts = (cnts / len(per_dat)) * 100
        dct_len = dict(zip(un, cnts))

        act_feat_idxs = [a for act in per_dat for a in list(act.keys())]
        uniq, counts = np.unique(act_feat_idxs, return_counts=True)
        counts = (counts / len(per_dat)) * 100
        dct = dict(zip(uniq, counts))
        print(dct)
        print("AVERAGE NUMBER OF FEATS INTERVENED")
        print(dct_len)
