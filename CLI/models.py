import re
import torch
from torch import nn, optim
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, mode: str, model_name: str):
        super(Model, self).__init__()

        self.mode = mode
        self.model_name = model_name

        assert re.match(r"^vgg$", model_name, re.IGNORECASE) or re.match(r"^resnet$", model_name, re.IGNORECASE) or re.match(r"^mobilenet$", model_name, re.IGNORECASE), "Invalid Model Name"

        if re.match(r"^vgg$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.encoder = models.vgg16_bn(pretrained=False, progress=True)    
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.encoder = models.vgg16_bn(pretrained=True, progress=True)
                self.freeze()
            self.encoder = nn.Sequential(*[*self.encoder.children()][:-2])
            self.encoder[0][0] = nn.Conv2d(in_channels=1, 
                                           out_channels=self.encoder[0][0].out_channels,
                                           kernel_size=self.encoder[0][0].kernel_size,
                                           stride=self.encoder[0][0].stride,
                                           padding=self.encoder[0][0].padding)
            fc = self.encoder[0][40].out_channels


        elif re.match(r"^resnet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.encoder = models.resnet50(pretrained=False, progress=True)  
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.encoder = models.resnet50(pretrained=True, progress=True)
                self.freeze()
            self.encoder = nn.Sequential(*[*self.encoder.children()][:-2])
            self.encoder[0] = nn.Conv2d(in_channels=1, 
                                        out_channels=self.encoder[0].out_channels,
                                        kernel_size=self.encoder[0].kernel_size,
                                        stride=self.encoder[0].stride,
                                        padding=self.encoder[0].padding)
            fc = self.encoder[7][2].conv3.out_channels              
        
        elif re.match(r"^mobilenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.encoder = models.mobilenet_v3_small(pretrained=False, progress=True) 
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.encoder = models.mobilenet_v3_small(pretrained=False, progress=True)
                self.freeze()
            self.encoder = nn.Sequential(*[*self.encoder.children()][:-2])
            self.encoder[0][0][0] = nn.Conv2d(in_channels=1, 
                                              out_channels=self.encoder[0][0][0].out_channels,
                                              kernel_size=self.encoder[0][0][0].kernel_size,
                                              stride=self.encoder[0][0][0].stride,
                                              padding=self.encoder[0][0][0].padding)
            fc = self.encoder[0][-1][0].out_channels

        self.decoder = nn.Sequential()
        self.decoder.add_module("DC1", nn.ConvTranspose2d(in_channels=fc, out_channels=512, kernel_size=4, stride=2, padding=1))
        self.decoder.add_module("AN1", nn.ReLU())
        self.decoder.add_module("DC2", nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2))
        self.decoder.add_module("AN2", nn.ReLU())
        self.decoder.add_module("DC3", nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2))
        self.decoder.add_module("AN3", nn.ReLU())
        self.decoder.add_module("DC4", nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2))
        self.decoder.add_module("AN4", nn.ReLU())
        self.decoder.add_module("DC5", nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=2))
        self.decoder.add_module("AN5", nn.Sigmoid())
                
    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False

        if re.match(r"^vgg$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*features.3[4-9].*", names, re.IGNORECASE) or re.match(r".*features.4[0-9].*", names, re.IGNORECASE) or re.match(r".*classifier.*", names, re.IGNORECASE):
                        params.requires_grad = True
        
        elif re.match(r"^resnet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*layer4.*", names, re.IGNORECASE):
                        params.requires_grad = True
        
        elif re.match(r"^mobilenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*features.9.*", names, re.IGNORECASE) or re.match(r".*features.1[0-2].*", names, re.IGNORECASE) or re.match(r".*classifier.*", names, re.IGNORECASE):
                        params.requires_grad = True

    def get_optimizer(self, lr: float = 1e-3, wd: float = 0.0):
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=lr, weight_decay=wd)
    
    def get_plateau_scheduler(self, optimizer=None, patience: int = 5, eps: float = 1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def get_model(seed: int, mode: str, model_name: str):
    torch.manual_seed(seed)
    model = Model(mode, model_name).to(DEVICE)

    return model



def get_model(seed: int, mode: str, model_name: str):
    torch.manual_seed(seed)
    model = Model(mode, model_name).to(DEVICE)

    return model
