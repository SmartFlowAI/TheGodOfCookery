from .config_test import *


def load_config(domain, key):
    return Config.get(domain).get(key, None)
