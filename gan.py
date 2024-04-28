import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets


class GAN(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module) -> None:
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def discriminator_loss(self, x, z):
        logit = self.discriminator(x)
        # log_D_x = torch.log(self.discriminator(x))
        log_D_x = F.logsigmoid(logit)

        with torch.no_grad():
            G_z = self.generator(z)
        log_1_minus_D_G_z = torch.log(1 - torch.sigmoid(self.discriminator(G_z)))

        discriminator_loss = -torch.mean(input=(log_D_x + log_1_minus_D_G_z), dim=0)

        return discriminator_loss

    def generator_loss(self, z, is_warmup: bool = False):
        G_z = self.generator(z) 
        # Note: if we're not careful, we'll accidentally get dloss/dparams for the discriminator
        # too because the discriminator is part of the generator loss!
        if is_warmup:
            # TODO - double check this against the Goodfellow paper
            generator_loss = -torch.log(torch.sigmoid(self.discriminator(G_z))).mean(dim=0)     # maximize log(D_g_z) objective
        else:
            generator_loss = torch.log(1 - torch.sigmoid(self.discriminator(G_z))).mean(dim=0)     # minimize (1 - D_g_z) loss

        return generator_loss

    def forward(self, z):
        return self.generator(z)

