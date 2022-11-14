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
from Discrminator_latent import Discriminator_l
from modules.generator_latent import Generator
#from Generator import Generator
from modules.model import Encoder,Decoder

import wandb
wandb.login(key='6b037e771b020c9d419f6468780b6a0640a9e9eb')


def train_fn(syn_encoder,real_encoder,disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler,epoch):
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
            fake_real = gen_H(syn)
            D_H_real = disc_H(real)
            D_H_fake = disc_H(fake_real.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_syn = gen_Z(real)
            D_Z_real = disc_Z(syn)
            D_Z_fake = disc_Z(fake_syn.detach())
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
            D_H_fake = disc_H(fake_real)
            D_Z_fake = disc_Z(fake_syn)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_syn = gen_Z(fake_real)
            cycle_real = gen_H(fake_syn)
            cycle_syn_loss = l1(syn, cycle_syn)
            cycle_real_loss = l1(real, cycle_real)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_syn = gen_Z(syn)
            identity_real = gen_H(real)
            identity_syn_loss = l1(syn, identity_syn)
            identity_real_loss = l1(real, identity_real)

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
        wandb.log({'H_real': H_reals/(idx+1),'H_fake': H_fakes/(idx+1), 'Epoch': epoch})
        loop.set_postfix(Epoch= epoch,H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))



def main():
    disc_H = Discriminator_l(config).to(config.DEVICE)
    disc_Z = Discriminator_l(config).to(config.DEVICE)
    gen_Z = Generator(config, num_residuals=5).to(config.DEVICE) #img_channels=3, num_residuals=9
    gen_H = Generator(config, num_residuals=5).to(config.DEVICE)
    syn_encoder = Encoder(config)
    real_encoder = Encoder(config)
    
    if config.load_syn:
        checkpoint1 = torch.load(config.checkpoint1,map_location='cuda')
        syn_encoder.load_state_dict(checkpoint1['model_state_dict'])
        syn_encoder.eval()
        syn_encoder.requires_grad_(False)
        syn_encoder = syn_encoder.to(config.DEVICE)
        print('syn_encoder',syn_encoder.training)
    
    if config.load_real:
        checkpoint2 = torch.load(config.checkpoint2,map_location='cuda')
        l = ['encoder']
        pretrained_dict = {x.replace('encoder.',''):y for x,y in checkpoint2['state_dict'].items() if x.split('.',1)[0] in l}
        real_encoder.load_state_dict(pretrained_dict)
        real_encoder.eval()
        real_encoder.requires_grad_(False)
        real_encoder = real_encoder.to(config.DEVICE)
        print('real_encoder',real_encoder.training)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = CustomDataset(
        root_real=r'/home/moravapa/Documents/Latent_DiscPrec/data/train1.txt', root_syn=r'/home/moravapa/Documents/Latent_DiscPrec/data/train2.txt', transform=config.transforms
    )
    '''val_dataset = HorseZebraDataset(root_horse="cyclegan_test/horse1", root_zebra="cyclegan_test/zebra1", transform=config.transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )'''
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
        train_fn(syn_encoder,real_encoder,disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,epoch)

        if epoch % 50 == 0 and config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    wandb.init(project="Syn_train", name="project", dir='Cycle_on_latent')
    wandb.run.name = 'Cycle_on_latent'
    main()
