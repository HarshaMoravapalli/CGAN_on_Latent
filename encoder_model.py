import torch
import torch.nn as nn

from modules.model import Encoder,Decoder
from modules.vec_quan import VectorQuantizer2 as VectorQuantizer

class Recon(nn.Module):
    def __init__(self, args):
        super(Recon, self).__init__()
        self.encoder = Encoder(args)
        #self.quantize = VectorQuantizer(args)
        #self.quant_conv = nn.Conv2d(args.latent_dim,args.latent_dim,1)

    def forward(self,image):

        i = self.encoder(image)
        #quant = self.quant_conv(i)
        #code_vec,code_loss, info = self.quantize(quant)
        return i

