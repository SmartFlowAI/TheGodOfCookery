import os
from omegaconf import OmegaConf
import json


def add_file_root_path(model_or_path: str, file_path_metas: dict, cfg={}):
    if isinstance(file_path_metas, dict):
        for k, v in file_path_metas.items():
            if isinstance(v, str):
                p = os.path.join(model_or_path, v)
                if os.path.exists(p):
                    cfg[k] = p
            elif isinstance(v, dict):
                if k not in cfg:
                    cfg[k] = {}
                add_file_root_path(model_or_path, v, cfg[k])

    return cfg


def get_local_model(model_path):
    kwargs = {}
    kwargs["model_path"] = model_path
    with open(os.path.join(model_path, "configuration.json"), 'r', encoding='utf-8') as f:
        conf_json = json.load(f)
        cfg = {}
        add_file_root_path(model_path, conf_json["file_path_metas"], cfg)
        cfg.update(kwargs)
        config = OmegaConf.load(cfg["config"])
        kwargs = OmegaConf.merge(config, cfg)
    kwargs["model"] = config["model"]
    return OmegaConf.to_container(kwargs, resolve=True)
