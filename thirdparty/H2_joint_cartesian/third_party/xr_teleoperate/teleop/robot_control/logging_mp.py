import logging


DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def basicConfig(*args, **kwargs):
    return logging.basicConfig(*args, **kwargs)


def getLogger(name=None):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger(name)
