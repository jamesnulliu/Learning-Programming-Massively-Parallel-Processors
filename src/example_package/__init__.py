import os
from os import path as osp
import torch

PACKAGE_DIR = osp.dirname(os.path.abspath(__file__))


def load_torch_ops(lib_dir: str):
    # @see: "https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.i5q3a2pv0qzc"
    for root, dirs, files in os.walk(lib_dir):
        for file in files:
            if file.endswith(".so") or file.endswith(".pyd"):
                torch.ops.load_library(os.path.join(root, file))


load_torch_ops(osp.join(PACKAGE_DIR, "_torch_ops"))
