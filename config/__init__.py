from .config import *

def load_config(domain, key):
    return Config.get(domain).get(key, None)