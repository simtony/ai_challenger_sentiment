# -*- coding: utf-8 -*-
from __future__ import division
import os
import sys
import argparse
import logging
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
from utils import ColoredFormatter, TFlogFilter, maybe_mkdir
import configuration
import runner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
default_work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
default_output_root = os.path.join(default_work_dir, "output")
default_data_dir = os.path.join(default_work_dir, "data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', help="Default work path", default=default_work_dir)
    parser.add_argument('--data_dir', help="Default data path", default=default_data_dir)
    parser.add_argument('--output_root', help="Default output path", default=default_output_root)
    parser.add_argument('--name', type=str, default='baseline')
    args, _ = parser.parse_known_args()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColoredFormatter(to_file=False))
    stream_handler.addFilter(TFlogFilter())
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    output_root = args.output_root
    name = args.name

    config = configuration.Configuration(name, args.work_dir, args.data_dir, args.output_root)
    config_parser = config.to_parser()
    update_args, _ = config_parser.parse_known_args()
    config.from_args(update_args)
    output_dir = maybe_mkdir(config.output_dir)
    log_path = os.path.join(output_dir, config.start_time + '.log')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ColoredFormatter(to_file=True))
    file_handler.addFilter(TFlogFilter())
    logger.addHandler(file_handler)
    try:
        config.log_params()
        runner.train(config, restore=False)
    except:
        logger.exception('Uncaught exception:')
        sys.exit(1)
