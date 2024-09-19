import os
import random
import numpy as np
import yaml
import logging


import torch


def set_configs(args):
    # Load default settings from YAML
    default_config = load_config(args.config)

    # Update arguments with defaults from YAML if not provided
    cfg = update_args(args, default_config)

    # Set up logging
    logger = setup_logger()
    cfg.logger = logger

    # Set random seed for reproducibility
    set_seed(cfg.seed)

    return cfg


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def update_args(args, default_config):
    args = _update_args(args, default_config)
    return _update_args_with_config(args, default_config)

def _update_args(args, default_config):
    """Update args with default config values where args are not provided."""
    for key in vars(args).keys():
        if getattr(args, key) is None:
            setattr(args, key, default_config[key])
    return args

def _update_args_with_config(args, default_config):
    """Update args with default config values where args are not provided."""
    for key, value in default_config.items():
        if not hasattr(args, key):
            # Add the key-value pair to args if it does not exist
            setattr(args, key, value)
        elif getattr(args, key) is None:
            # Update existing key if it is None
            setattr(args, key, value)
    return args


def setup_logger(file_path):
    """Set up logging configuration."""
    logger = logging.getLogger('PyTorchLogger')
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create a file handler for writing log messages to a file
    file_handler = logging.FileHandler(f'{file_path}/training.log')
    file_handler.setLevel(logging.INFO)

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define the format of log messages
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger