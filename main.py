from generator import Generator, DCGenerator
from discriminator import Discriminator, DCDiscriminator
from gan import GAN
from wgan import WGAN
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


# Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 30
BATCH_SIZE = 128
GAN_LR_RATE = 3e-3
WGAN_LR_RATE = 0.00005
LATENT_DIM_NOISE = 64
IMAGE_INPUT_DIM = 784                                                   # 28 x 28
K = 5                                                                  # as mentioned in the paper
CLIPPING_CONSTANT = 0.001

def train_gan_model(train_loader, gan: GAN, optimizer_generator, 
                optimizer_discriminator, z_noise, writer = None):
    disc_loss_list = []
    gen_loss_list = []
    step = 0
    is_warmup = True
    print("Vanilla GAN training starts...\n")
    print()

    for epoch in tqdm(range(NUM_EPOCHS), desc = "EPOCHS"):
            
        for batch_idx, (x, label) in enumerate(tqdm(train_loader, desc = "Batches", leave = False)):
            step += 1

            for _ in range(K):
        
            
                x = x.view(-1, 784).to(DEVICE)
                batch_size = x.shape[0]
                noise = torch.randn(batch_size, LATENT_DIM_NOISE).to(DEVICE)

                disc_loss = gan.discriminator_loss(x, noise)
                disc_loss_list.append(disc_loss.item())

                optimizer_discriminator.zero_grad()
                disc_loss.backward()
                optimizer_discriminator.step()




            gen_loss = gan.generator_loss(noise, is_warmup=True)
            gen_loss_list.append(gen_loss.item())

            optimizer_generator.zero_grad()
            gen_loss.backward()
            # Unnecessary, paranoid book-keeping. Make sure that after we call generator.backward(),
            # all of the discriminator params' .grad attribute is zero.
            optimizer_discriminator.zero_grad()
            optimizer_generator.step()

            if writer is not None:
                writer.add_scalar("loss/discriminator", disc_loss.item(), global_step=step)
                writer.add_scalar("loss/generator", gen_loss.item(), global_step=step)
                for i, g_param in enumerate(gan.generator.parameters()):
                    writer.add_scalar(f"weights/generator/{i}", torch.sum(g_param**2).sqrt(), step)
                for i, d_param in enumerate(gan.discriminator.parameters()):
                    writer.add_scalar(f"weights/discriminator/{i}", torch.sum(d_param**2).sqrt(), step)
            
            if torch.isnan(gen_loss + disc_loss):
                raise RuntimeError("NaN encountered.")
        
        # After each epoch, generate some images
        if writer is not None:
            with torch.no_grad():
                output = gan(z_noise).view(-1, 1, 28, 28)
                grid = torchvision.utils.make_grid(output)
                writer.add_image('generated images', grid, 0)
        
        # After each epoch, save a checkpoint... TODO


    print()
    print("Vanilla GAN training is done")

    # print(output)

    # save the model
    checkpoint = {'generator_state_dict': gan.generator.state_dict(),
                    'discriminator_state_dict': gan.discriminator.state_dict(),
                    'generator_optimizer_state_dict': optimizer_generator.state_dict(),
                    'discriminator_optimizer_state_dict': optimizer_discriminator.state_dict(),
                    'epoch': epoch}
    torch.save(checkpoint, 'logs/gan/checkpoint.pt')


    # save loss graphs for generator and discriminator
    plt.figure()
    plt.title('GAN Generator Loss Graph ')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.plot(range(step),  gen_loss_list, color = 'blue')
    plt.savefig('plots/gan/Generator_loss.png')
    plt.close()

    plt.figure()
    plt.title('GAN Discriminator Loss Graph ')
    plt.xlabel('steps x K')
    plt.ylabel('loss')
    plt.plot(range(step * K),  disc_loss_list, color = 'red')
    plt.savefig('plots/gan/Discriminator_loss.png')
    plt.close()

    return gan


def train_wgan_model(train_loader, wgan, optimizer_generator, optimizer_discriminator, z_noise):
    disc_loss_list = []
    gen_loss_list = []
    step = 0

    print("Wasserstein GAN training starts...")

    for epoch in tqdm(range(NUM_EPOCHS), desc = "EPOCHS"):
            
        for batch_idx, (x, label) in enumerate(tqdm(train_loader, desc = "Batches", leave = False)):
            step += 1

            for _ in range(K):

                # x = x.view(-1, 784).to(DEVICE)
                # batch_size = x.shape[0]
                x = x.to(DEVICE)
                noise = torch.randn(BATCH_SIZE, LATENT_DIM_NOISE, x.shape[1]).to(DEVICE)
                noise = torch.randn(BATCH_SIZE, LATENT_DIM_NOISE, 28, 28 ).to(DEVICE)


                # noise = noise.unsqueeze(0)

                disc_loss = wgan.discriminator_loss(x, noise)
                disc_loss_list.append(disc_loss.item())

                optimizer_discriminator.zero_grad()
                disc_loss.backward()
                optimizer_discriminator.step()

                # clip weights between -c to c
                wgan.dcdiscriminator.weight_clip(CLIPPING_CONSTANT)

            gen_loss = wgan.generator_loss(noise)
            gen_loss_list.append(gen_loss.item())

            optimizer_generator.zero_grad()
            gen_loss.backward()
            optimizer_generator.step()
    
    print("Wasserstein GAN training Ends...")
    print()

    checkpoint = {'generator_state_dict': wgan.dcgenerator.state_dict(),
                    'discriminator_state_dict': wgan.dcdiscriminator.state_dict(),
                    'generator_optimizer_state_dict': optimizer_generator.state_dict(),
                    'discriminator_optimizer_state_dict': optimizer_discriminator.state_dict(),
                    'epoch': epoch}
    torch.save(checkpoint, 'logs/wgan/checkpoint.pt')

    plt.figure()
    plt.title('Wasserstein GAN Generator Loss Graph')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.plot(range(step),  gen_loss_list, color = 'blue')
    plt.savefig('plots/wgan/Generator_loss.png')
    plt.close()

    plt.figure()
    plt.title('Wasserstein GAN Discriminator Loss Graph')
    plt.xlabel('steps x K')
    plt.ylabel('loss')
    plt.plot(range(step * K),  disc_loss_list, color = 'red')
    plt.savefig('plots/wgan/Discriminator_loss.png')
    plt.close()

    return wgan


def train_gan(train_loader):
    discriminator = Discriminator(img_dim_ip=IMAGE_INPUT_DIM).to(DEVICE)
    generator = Generator(img_dim_op=IMAGE_INPUT_DIM , z_dim_ip=LATENT_DIM_NOISE).to(DEVICE)
    

    z_noise = torch.randn((BATCH_SIZE, LATENT_DIM_NOISE)).to(DEVICE)

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr= GAN_LR_RATE, weight_decay=1e-4)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=GAN_LR_RATE, weight_decay=1e-4)

    if 'checkpoint.pt' in os.listdir('logs/gan/'):
        gan = GAN(generator, discriminator)

        checkpoint = torch.load('logs/gan/checkpoint.pt', map_location='cpu')
        gan.generator.load_state_dict(checkpoint["generator_state_dict"])
        gan.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_generator.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        optimizer_discriminator.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        gan = GAN(generator, discriminator)

    writer = SummaryWriter('logs/gan/' + str(datetime.datetime.now()))

    gan = train_gan_model(train_loader, gan, optimizer_generator, optimizer_discriminator, z_noise, writer=writer)

    # check and save output
    gan.cpu()
    gan.eval()
    n_samples = 100  # Number of samples to save
    z_noise = torch.randn((BATCH_SIZE, LATENT_DIM_NOISE))
    fig, axs = plt.subplots(1, n_samples, figsize=(10, 10))
    with torch.no_grad():
        for i in range(n_samples):
            x = gan( z_noise )

    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(x[i*10+j].reshape(28,28))
            axs[i, j].axis('off')
    fig.tight_layout()
    plt.show()
    plt.savefig('output/gan_generated_samples.png')
        

def train_wgan(train_loader):
    discriminator = DCDiscriminator(input_img_x_dim=1).to(DEVICE)
    generator = DCGenerator(img_dim=IMAGE_INPUT_DIM , z_ip_dim=LATENT_DIM_NOISE).to(DEVICE)
    wgan = WGAN(generator, discriminator)

    z_noise = torch.randn((BATCH_SIZE, LATENT_DIM_NOISE)).to(DEVICE)

    optimizer_generator = torch.optim.RMSprop(generator.parameters(), lr= WGAN_LR_RATE, weight_decay=1e-5)
    optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=WGAN_LR_RATE, weight_decay=1e-5)

    trained_wgan = train_wgan_model(train_loader, wgan, optimizer_generator, optimizer_discriminator, z_noise)

    trained_wgan.cpu()
    trained_wgan.eval()
    n_samples = 100  # Number of samples to save
    z_noise = torch.randn((BATCH_SIZE, LATENT_DIM_NOISE, 28, 28))
    fig, axs = plt.subplots(1, n_samples, figsize=(10, 10))
    with torch.no_grad():
        for i in range(n_samples):
            x = trained_wgan( z_noise )
            # axs[i].imshow(x[i].reshape(28, 28), cmap='gray')
            # x_list.append(x)

    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(x[i*10+j].reshape(28,28))
            axs[i, j].axis('off')
    fig.tight_layout()
    plt.show()
    plt.savefig('output/wgan_generated_samples.png')


def main():
    dataset = datasets.MNIST(root='dataset/', train=True, transform = transforms.Compose([
                                                                 transforms.ToTensor()
                                                                ]), download=True)
    
    train_loader = DataLoader(dataset=dataset, batch_size= BATCH_SIZE, shuffle=True)

    
    # train_gan(train_loader)

    train_wgan(train_loader)



if __name__ == "__main__":
    main()