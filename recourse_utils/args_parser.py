import argparse
import ast
import os
import pickle

import yaml
import glob
from yacs.config import CfgNode as CN


def list_intersection(l1, l2):
    out = list(set(l1) & set(l2))
    if len(out) > 0:
        my_type = type(out[0])
        assert all(isinstance(x, my_type) for x in out)
    return out


def list_union(l1, l2):
    out = list(set(l1) | set(l2))
    if len(out) > 0:
        my_type = type(out[0])
        assert all(isinstance(x, my_type) for x in out)
    return out


def list_substract(l, l_substact):
    out = list(set(l) - set(l_substact))
    if len(out) > 0:
        my_type = type(out[0])
        assert all(isinstance(x, my_type) for x in out)
    return out


def to_str(elem):
    if isinstance(elem, list):
        return '_'.join([str(s) for s in elem])
    else:
        return str(elem)


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split("+"):
            k, v = kv.split("=")
            if isinstance(v, str):
                if k not in ['missing_perc', 'features_s', 'features_e', 'features_l']:
                    try:
                        v = ast.literal_eval(v)
                    except:
                        pass
                else:
                    try:
                        vi = ast.literal_eval(v)
                    except:
                        vi = 2
                        pass
                    v = vi if vi < 1.0 else v

            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def newest(path, include_last=False):
    if not os.path.exists(path):
        return None
    files = os.listdir(path)
    if len(files) == 0:
        return None
    paths = []
    for basename in files:
        if 'last.ckpt' not in basename:
            paths.append(os.path.join(path, basename))
        else:
            if include_last:
                paths.append(os.path.join(path, basename))

    return max(paths, key=os.path.getctime)


def save_yaml(yaml_object, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(yaml_object, yaml_file, default_flow_style=False)

    if "verbose" in yaml_object.keys() and yaml_object["verbose"]:
        print(f'Saving yaml: {file_path}')
    return


def save_obj(filename_no_extension, obj, ext='.pkl'):
    with open(filename_no_extension + ext, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def to_yacs(yacs_cfg, cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            yv = CN()
            yacs_cfg[k] = to_yacs(yv, v)
        else:
            yacs_cfg[k] = v
    return yacs_cfg


def parse_args(yaml_file):

    yaml_files = glob.glob(yaml_file)
    cfg = None
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    assert cfg is not None, f"didnt find the yaml file {yaml_file}"
    return cfg


def flatten_cfg(cfg):
    cfg_flat = {}
    for key, value in cfg.items():
        if not isinstance(value, dict):
            cfg_flat[key] = value
        else:
            for key2, value2 in value.items():
                if not isinstance(value2, dict):
                    cfg_flat[f'{key}_{key2}'] = value2
                else:
                    for key3, value3 in value2.items():
                        cfg_flat[f'{key}_{key2}_{key3}'] = value3

    return cfg_flat


def get_experiment_folder_policy(cfg, keys1_string, dict2, dict3):
    keys2_string, keys3_string = '', ''
    # if dict1 is not None:
    #     keys1_string = '_'.join([f"{to_str(v)}" for k, v in dict1.items()])
    #     keys1_string = '_' + keys1_string

    if dict2 is not None:
        keys2_string = '_'.join([f"{to_str(v)}" for k, v in dict2.items()])
        keys2_string = '_' + keys2_string

    if dict3 is not None:
        keys3_string = '_'.join([f"{to_str(v)}" for k, v in dict3.items()])
        keys3_string = '_' + keys3_string

    # keys1 = cfg['classifier']['name'] + keys1_string
    keys2 = cfg['meta']['name'] + keys2_string
    keys3 = cfg['predictor']['name'] + keys3_string

    return os.path.join(
        f"exp_{cfg['dataset']['name']}_{cfg['dataset']['params']['equations_type']}_{cfg['dataset']['params']['meta_sep_data']}",
        f"clf_{keys1_string}",
        keys2, keys3)


def get_experiment_folder(cfg, dict1, dict2, dict3):
    raise NotImplementedError
