import yaml
from typing import Dict, Tuple
from types import SimpleNamespace

import os
import torch
import numpy as np
import pandas as pd
import random
import pytorch_lightning as pl

def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    pl.seed_everything(seed, workers=True, verbose=False)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_config(config_fp: str) -> Dict:
    
    with open(config_fp, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    
    if "data" in config and "seq_lmdb" in config["data"]:
        d_seq_m_s = config["data"]["seq_lmdb"]["map_size"]
        if isinstance(d_seq_m_s, str):
            import ast, operator
            def _safe_eval_numeric(expr):
                _OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
                        ast.Mult: operator.mul, ast.Pow: operator.pow}
                def _ev(node):
                    if isinstance(node, ast.Expression): return _ev(node.body)
                    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                        return node.value
                    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
                    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                        return -_ev(node.operand)
                    raise ValueError(f"Unsupported expression: {ast.dump(node)}")
                return _ev(ast.parse(expr, mode='eval'))
            config["data"]["seq_lmdb"]["map_size"] = _safe_eval_numeric(d_seq_m_s)
    
    return config


def df_split(df: pd.DataFrame, ratio: Tuple[float, float, float], seed: int, shuffle: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed=seed)

    n_samples = df.shape[0]
    original_indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(original_indices)

    train_size = int(np.round(n_samples * ratio[0]))
    valid_size = int(np.round(n_samples * ratio[1]))
    test_size = n_samples - train_size - valid_size
    
    train_indices = original_indices[:train_size]
    valid_indices = original_indices[train_size:train_size+valid_size]
    test_indices = original_indices[train_size+valid_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    return train_df, valid_df, test_df