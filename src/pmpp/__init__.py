from pathlib import Path
import torch

PACKAGE_DIR = Path(__file__).parent.absolute()


def load_torch_ops(lib_dir: Path):
    # @see: "https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.i5q3a2pv0qzc"
    for root, dirs, files in lib_dir.walk():
        for file in files:
            if file.endswith(".so") or file.endswith(".pyd"):
                torch.ops.load_library(root / file)


load_torch_ops(PACKAGE_DIR / "_torch_ops")