import os
import sys
import numpy as np

from CLI.utils import breaker

def main():
    args: tuple = ("--path", "-p")
    path: str = "data"
    if args in sys.argv: path = sys.argv[sys.argv.index(args) + 1]

    assert "bw_images.npy" in os.listdir(path) and "ab_images.npy" in os.listdir(path), "Please run python np_make.py"

    bw_images = np.load(os.path.join(path, "bw_images.npy"))
    ab_images = np.load(os.path.join(path, "ab_images.npy"))

    breaker()
    print(f"BW Images Mean : {bw_images.mean() / 255}")
    print(f"BW Images Std  : {bw_images.std() / 255}")
    breaker()
    print(f"A* Images Mean : {ab_images[:, :, 0].mean() / 255}")
    print(f"A* Images Std  : {bw_images[:, :, 0].std() / 255}")
    breaker()
    print(f"B* Images Mean : {ab_images[:, :, 1].mean() / 255}")
    print(f"B* Images Std  : {ab_images[:, :, 1].std() / 255}")
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
