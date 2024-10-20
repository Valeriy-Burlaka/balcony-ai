import logging
from logging.handlers import RotatingFileHandler

loggers = {}

LOGGER_FORMAT = "%(name)s - %(asctime)s : [%(levelname)s] %(message)s"

def map_verbosity_level_to_logging_level(verbosity: int) -> int:
    level_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    return level_map.get(verbosity, logging.DEBUG)

def get_logger(name: str, verbosity: int, log_dir="tmp/logs"):
    if name in loggers:
        return loggers[name]

    logger = logging.getLogger(name)
    logger.propagate = False
    # Set how texts are exposed to the screen according to the selected verbosity level.
    # This is a simplistic interpretation of the Canonocal CLI guidelines
    #   (see https://discourse.ubuntu.com/t/cli-verbosity-levels/26973).
    level = map_verbosity_level_to_logging_level(verbosity)

    formatter = logging.Formatter(LOGGER_FORMAT)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)
    logger.addHandler(ch)

    fh = RotatingFileHandler(f"./{log_dir}/{name}.log", maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    loggers[name] = logger

    return logger

def update_app_verbosity_level(verbosity: int):
    level = map_verbosity_level_to_logging_level(verbosity)
    logging.basicConfig(level=level)

def update_logger_verbosity_level(logger: logging.Logger, verbosity: int):
    level = map_verbosity_level_to_logging_level(verbosity)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
