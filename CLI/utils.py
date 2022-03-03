
import os
import re
import cv2
import json
import torch
import imgaug
import numpy as np
import matplotlib.pyplot as plt

from time import time
from imgaug import augmenters
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from sklearn.model_selection import KFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM_BW = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49853676753690856], [0.2798434312494038])])
TRANSFORM_A = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5123545469789598], [0.0520517560541576])])
TRANSFORM_B = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5123588190530597], [0.05202467833992707])])


SAVE_PATH = "saves"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


class DS(Dataset):
    def __init__(self, bw_images: np.ndarray, bw_transform=None, ab_images: np.ndarray = None, ab_transform=None, mode: str = "train"):

        assert re.match(r"^train$", mode, re.IGNORECASE) or re.match(r"^valid$", mode, re.IGNORECASE) or re.match(r"^test$", mode, re.IGNORECASE), "Invalid Mode"
        
        self.mode = mode
        self.bw_transform = bw_transform
        self.bw_images = bw_images

        if re.match(r"^train$", mode, re.IGNORECASE) or re.match(r"^valid$", mode, re.IGNORECASE):
            self.ab_transform = ab_transform
            self.ab_images = ab_images
            
    def __len__(self):
        return self.bw_images.shape[0]

    def __getitem__(self, idx):
        if re.match(r"^train$", self.mode, re.IGNORECASE) or re.match(r"^valid$", self.mode, re.IGNORECASE):
            return self.bw_transform(self.bw_images[idx]), self.ab_transform(self.ab_images[idx])
        else:
            return self.bw_transform(self.bw_images[idx])


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def load_data(path: str, b_channel: bool) -> tuple:
    assert "bw_images.npy" in os.listdir(path) and "ab_images.npy" in os.listdir(path), "Please run python np_make.py"

    bw_images = np.load(os.path.join(path, "bw_images.npy"))
    ab_images = np.load(os.path.join(path, "ab_images.npy"))

    if b_channel:
        # return bw_images, ab_images[:, :, :, 1]
        return bw_images, ab_images
    else:
        # return bw_images, ab_images[:, :, :, 0]
        return bw_images, ab_images

        



def get_augment(seed: int):
    imgaug.seed(seed)
    augment = augmenters.Sequential([
        augmenters.HorizontalFLip(p=0.15),
        augmenters.VerticalFLip(p=0.15),
        augmenters.Affine(scale=(0.5, 1.5), translate_percent=(-0.1, 0.1), rotate=(-45, 45)),
    ])
    return augment


def prepare_train_and_valid_dataloaders(path: str, mode: str, b_channel: bool, batch_size: int, seed: int, augment: bool=False):

    bw_images, ab_images = load_data(path, b_channel)

    for tr_idx, va_idx in KFold(n_splits=5, shuffle=True, random_state=seed).split(bw_images):
        tr_bw_images, va_bw_images, tr_ab_images, va_ab_images = bw_images[tr_idx], bw_images[va_idx], ab_images[tr_idx], ab_images[va_idx]
        break

    if augment:
        augmenter = get_augment(seed)
        tr_bw_images = augmenter(images=tr_bw_images)
        tr_ab_images = augmenter(images=tr_ab_images)

    if b_channel:
        tr_data_setup = DS(tr_bw_images, TRANSFORM_BW, tr_ab_images, TRANSFORM_B, "train")
        va_data_setup = DS(va_bw_images, TRANSFORM_BW, va_ab_images, TRANSFORM_B, "valid")
    else:
        tr_data_setup = DS(tr_bw_images, TRANSFORM_BW, tr_ab_images, TRANSFORM_A, "train")
        va_data_setup = DS(va_bw_images, TRANSFORM_BW, va_ab_images, TRANSFORM_A, "valid")

    dataloaders = {
        "train" : DL(tr_data_setup, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(seed)),
        "valid" : DL(va_data_setup, batch_size=batch_size, shuffle=False)
    }

    return dataloaders


def save_graphs(L: list) -> None:
    TL, VL = [], []
    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
    x_Axis = np.arange(1, len(TL) + 1)
    plt.figure("Plots")
    plt.plot(x_Axis, TL, "r", label="Train")
    plt.plot(x_Axis, VL, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Loss Graph")
    plt.savefig(os.path.join(SAVE_PATH, "Graphs.jpg"))
    plt.close("Plots")


def fit(model=None, optimizer=None, scheduler=None, epochs=None, early_stopping_patience=None, dataloaders=None, verbose=False):
    
    if verbose:
        breaker()
        print("Training ...")
        breaker()

    bestLoss = {"train" : np.inf, "valid" : np.inf}
    Losses = []
    name = "state.pt"

    start_time = time()
    for e in range(epochs):
        e_st = time()
        epochLoss = {"train" : 0.0, "valid" : 0.0}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            lossPerPass = []

            for X, y in dataloaders[phase]:
                X, y = X.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    _, output = model(X)
                    loss = torch.nn.MSELoss()(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item())
            epochLoss[phase] = np.mean(np.array(lossPerPass))
        Losses.append(epochLoss)

        if early_stopping_patience:
            if epochLoss["valid"] < bestLoss["valid"]:
                bestLoss = epochLoss
                BLE = e + 1
                torch.save({"model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict()},
                           os.path.join(SAVE_PATH, name))
                early_stopping_step = 0
            else:
                early_stopping_step += 1
                if early_stopping_step > early_stopping_patience:
                    print("\nEarly Stopping at Epoch {}".format(e + 1))
                    break
        
        if epochLoss["valid"] < bestLoss["valid"]:
            bestLoss = epochLoss
            BLE = e + 1
            torch.save({"model_state_dict" : model.state_dict(),
                        "optim_state_dict" : optimizer.state_dict()},
                        os.path.join(SAVE_PATH, name))
        
        if scheduler:
            scheduler.step(epochLoss["valid"])
        
        if verbose:
            print("Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} | Time: {:.2f} seconds".format(e+1, epochLoss["train"], epochLoss["valid"], time()-e_st))

    if verbose:                                           
        breaker()
        print(f"Best Validation Loss at Epoch {BLE}")
        breaker()
        print("Time Taken [{} Epochs] : {:.2f} minutes".format(len(Losses), (time()-start_time)/60))
        breaker()
        print("Training Completed")
        breaker()

    return Losses, BLE, name


def predict(model=None, mode: str = None, image_path: str = None, size: int = 320) -> np.ndarray:
    model.load_state_dict(torch.load(os.path.join(SAVE_PATH, "state.pt"), map_location=DEVICE)["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    bw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = bw_image.shape
    bw_image = cv2.resize(src=bw_image, dsize=(size, size), interpolation=cv2.INTER_AREA).reshape(size, size, 1)

    with torch.no_grad():
        _, ab_image = model(TRANSFORM_BW(bw_image).to(DEVICE).unsqueeze(dim=0))

    ab_image = ab_image.squeeze()
    ab_image = ab_image.detach().cpu().numpy().transpose(1, 2, 0)
    ab_image = np.clip((ab_image * 255), 0, 255).astype("uint8")

    color_image = np.concatenate((bw_image, ab_image), axis=2)

    return cv2.resize(src=color_image, dsize=(w, h), interpolation=cv2.INTER_AREA)


def show(image: np.ndarray, title: bool = False) -> None:
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()
