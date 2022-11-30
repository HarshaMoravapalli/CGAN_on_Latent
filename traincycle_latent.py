import torch
from dataset import CustomDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Discriminator_latent import Discriminator
from modules.generator_latent import Generator
#from Generator import Generator
from modules.model import Encoder,Decoder

import wandb
wandb.login(key='6b037e771b020c9d419f6468780b6a0640a9e9eb')


def train_fn(syn_encoder,real_encoder,disc_real, disc_fake, gen_fake, gen_real, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler,epoch):
    H_reals = 0
    H_fakes = 0
    total_samples = len(loader)
    loop = tqdm(loader, leave=True)
    
    for idx, (syn, real) in enumerate(loop):
        syn_image = syn.to(config.DEVICE, dtype=torch.float)
        real_image = real.to(config.DEVICE, dtype=torch.float)
        syn = syn_encoder(syn_image)
        real = real_encoder(real_image)
        with torch.cuda.amp.autocast():
            fake_real = gen_real(syn)
            D_H_real = disc_real(real)
            D_H_fake = disc_real(fake_real.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_syn = gen_fake(real)
            D_Z_real = disc_fake(syn)
            D_Z_fake = disc_fake(fake_syn.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_real(fake_real)
            D_Z_fake = disc_fake(fake_syn)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_syn = gen_fake(fake_real)
            cycle_real = gen_real(fake_syn)
            cycle_syn_loss = l1(syn, cycle_syn)
            cycle_real_loss = l1(real, cycle_real)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            '''identity_syn = gen_fake(syn)
            identity_real = gen_real(real)
            identity_syn_loss = l1(syn, identity_syn)
            identity_real_loss = l1(real, identity_real)'''

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_syn_loss * config.LAMBDA_CYCLE
                + cycle_real_loss * config.LAMBDA_CYCLE
                + identity_real_loss * config.LAMBDA_IDENTITY
                + identity_syn_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        wandb.log({'D_loss': D_loss,'G_Loss': G_loss,'D_real':H_reals/(idx+1), 'D_fake':H_fakes/(idx+1), 'Epoch': epoch})
        loop.set_postfix(Epoch= epoch,D_loss=D_loss, G_loss=G_loss)



def main():
    disc_real = Discriminator().to(config.DEVICE)
    disc_fake = Discriminator().to(config.DEVICE)
    gen_fake = Generator(config, num_residuals=5).to(config.DEVICE) #img_channels=3, num_residuals=9
    gen_real = Generator(config, num_residuals=5).to(config.DEVICE)
    syn_encoder = Encoder(config)
    real_encoder = Encoder(config)
    
    if config.load_syn:
        checkpoint1 = torch.load(config.checkpoint1,map_location='cuda')
        l1 = ['encoder']
        pretrained_dict1 = {x.replace('encoder.',''):y for x,y in checkpoint1['model_state_dict'].items() if x.split('.',1)[0] in l1}
        syn_encoder.load_state_dict(pretrained_dict1)
        syn_encoder.eval()
        syn_encoder.requires_grad_(False)
        syn_encoder = syn_encoder.to(config.DEVICE)
        print('syn_encoder',syn_encoder.training)
    
    if config.load_real:
        checkpoint2 = torch.load(config.checkpoint2,map_location='cuda')
        l = ['encoder']
        pretrained_dict = {x.replace('encoder.',''):y for x,y in checkpoint2['model_state_dict'].items() if x.split('.',1)[0] in l}
        real_encoder.load_state_dict(pretrained_dict)
        real_encoder.eval()
        real_encoder.requires_grad_(False)
        real_encoder = real_encoder.to(config.DEVICE)
        print('real_encoder',real_encoder.training)

    opt_disc = optim.Adam(
        list(disc_real.parameters()) + list(disc_fake.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_fake.parameters()) + list(gen_real.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()


    dataset = CustomDataset(
        root_real=r'/home/moravapa/Documents/Taming/data/train_1.txt', root_syn=r'/home/moravapa/Documents/Taming/data/train_syn.txt', transform=config.transforms
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(syn_encoder,real_encoder,disc_real, disc_fake, gen_fake, gen_real, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,epoch)

        if epoch % 100 == 0 and config.SAVE_MODEL:
            torch.save(gen_real.state_dict(), config.CHECKPOINT_gen_real)
            torch.save(gen_fake.state_dict(), config.CHECKPOINT_gen_fake)

        

if __name__ == "__main__":
    wandb.init(project="Experiments", name="project", dir='Experiments')
    wandb.run.name = 'Cycle_on_latent'
    main()
