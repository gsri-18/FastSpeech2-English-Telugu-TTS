""" Config loading shared by the English/ and Telugu/ scripts.

Each language has one config.yaml with `preprocess`, `model` and `train`
sections, which replaces upstream's three yaml files per dataset.
"""

import os

import yaml


def _resolve(paths, base):
    return {
        k: (os.path.normpath(os.path.join(base, v)) if isinstance(v, str) else v)
        for k, v in paths.items()
    }


def load_config(path):
    """Read a config.yaml and return the upstream-style
    (preprocess_config, model_config, train_config) triple. Every entry under a
    `path` block is resolved relative to the directory holding config.yaml."""
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    preprocess_config, model_config, train_config = (
        cfg["preprocess"], cfg["model"], cfg["train"]
    )
    preprocess_config["path"] = _resolve(preprocess_config["path"], base)
    train_config["path"] = _resolve(train_config["path"], base)
    return preprocess_config, model_config, train_config
