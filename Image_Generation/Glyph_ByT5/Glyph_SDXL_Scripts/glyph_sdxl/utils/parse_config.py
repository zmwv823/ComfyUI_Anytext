import argparse
import os
import os.path as osp
from mmengine.config import Config


def parse_config(path=None):
    if path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_dir', type=str)
        args = parser.parse_args()
        path = args.config_dir
    config = Config.fromfile(path)
    
    config.config_dir = path

    return config
