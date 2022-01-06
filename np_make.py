import os
import sys
import cv2
import numpy as np

from CLI.utils import breaker


def reserve_memory(path: str, size: int) -> tuple:
    total_num_files: int = 0
    for name in os.listdir(path):
        if name == "test":
            pass
        else:
            total_num_files += len(os.listdir(os.path.join(path, name)))
    bw_images  = np.zeros((total_num_files, size, size, 1), dtype=np.uint8)
    ab_images = np.zeros((total_num_files, size, size, 2), dtype=np.uint8)
    return bw_images, ab_images


def preprocess(image: np.ndarray, size: int) -> tuple:
    image = cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2LAB)
    return image[:, :, 0].reshape(size, size, 1), image[:, :, 1:]


def main() -> None:
    args_1: tuple = ("--path", "-p")
    args_2: tuple = ("--size", "-s")
    args_3: tuple = ("--kaggle", "-k")

    path: str = "data"
    size: int = 320
    kaggle: bool = False

    if args_1[0] in sys.argv: path = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: path = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: size = int(sys.argv[sys.argv.index(args_2[0]) + 1])
    if args_2[1] in sys.argv: size = int(sys.argv[sys.argv.index(args_2[1]) + 1])

    if args_3[0] in sys.argv or args_3[1] in sys.argv: kaggle = True

    assert os.path.exists(path), "Please Unzip the data first using python unzip.py"

    breaker()
    print("Reserving Memory ...")

    bw_images, ab_images = reserve_memory(path, size)

    i: int = 0
    j: int = 0

    breaker()
    print("Saving Images to Numpy Arrays ...")

    folders = sorted([folder_name for folder_name in os.listdir(path) if folder_name != "test"])

    for folder_name in folders:
        for filename in os.listdir(os.path.join(path, folder_name)):
            image = cv2.imread(os.path.join(os.path.join(path, folder_name), filename), cv2.IMREAD_COLOR)

            bw_images[i], ab_images[i] = preprocess(image, size)

            i += 1

    if kaggle:
        np.save("./bw_images.npy", bw_images)
        np.save("./ab_images.npy", ab_images)
    else:
        np.save(os.path.join(path, "bw_images.npy"), bw_images)
        np.save(os.path.join(path, "ab_images.npy"), ab_images)

    breaker()
    print("Saving Complete")
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
