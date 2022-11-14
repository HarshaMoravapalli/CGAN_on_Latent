import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 501
in_channels = 128
out_channels = 3
ch_mult = [1,1,2,2,4]
resolution = 256
image_channels = 3
num_codebook_vectors = 1024
latent_dim = 256
image_size = 256
dropout = 0.0
load_syn = False
load_real = True

checkpoint1 = r'/home/moravapa/Documents/ckpt/perc_400.ckpt'
checkpoint2 = r"/home/moravapa/Documents/ckpt/last.ckpt"
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "saved_images/ckpt/generator_lr1.pth"
CHECKPOINT_GEN_Z = "saved_images/ckpt/generator_lu1.pth"
CHECKPOINT_CRITIC_H = "saved_images/ckpt/critich_lr1.pth"
CHECKPOINT_CRITIC_Z = "saved_images/ckpt/criticz_lu1.pth"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
