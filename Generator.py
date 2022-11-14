import torch
import torch.nn as nn
import argparse
import config

from modules.model import Encoder,Decoder

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    def forward(self,image):
        enc_img = self.encoder(image)
        dec_img = self.decoder(enc_img)
        return dec_img

