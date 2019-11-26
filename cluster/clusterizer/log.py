import os
import logging

LOGGER_ROOT_DIR = "."


def get_logger(name, level=logging.INFO):  # TODO cache returned value with name+level as cache key
    logger = logging.getLogger(name)
    logger.setLevel(level)
    root_dir = os.path.abspath(LOGGER_ROOT_DIR)
    logger_path = os.path.join(root_dir, f"{name}.log")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    fh = logging.FileHandler(logger_path)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    return logger
