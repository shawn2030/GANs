import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets


class WGAN(nn.Module):
    def __init__(self, dcgenerator: nn.Module, dcdiscriminator: nn.Module) -> None:
        super(WGAN, self).__init__()
        self.dcgenerator = dcgenerator
        self.dcdiscriminator = dcdiscriminator

    # generator loss is the same as a simple gan
    def generator_loss(self, z):
        G_z = self.dcgenerator(z)

        G_z = G_z.reshape(-1, 1, 28, 28)
        # TODO - update to proper WGAN generator loss
        # generator_loss = torch.log(1 - torch.sigmoid(self.dcdiscriminator(G_z))).mean(dim=0)     # minimize (1 - D_g_z) loss
        generator_loss = -self.dcdiscriminator(G_z).mean(dim=0)     

        return generator_loss


    # discriminator loss changes w.r.t simple gan,
    # this becomes a continuous problem now with parameter W.
    def discriminator_loss(self,x, z):
        # z shape ~ [128, 64, 28, 28]
        # x shape ~ [128, 1, 28, 28]

        with torch.no_grad():
            G_z = self.dcgenerator(z)           

        G_z = G_z.reshape(-1, 1, 28, 28)

        # D_x = torch.sigmoid(self.dcdiscriminator(x))  # size (len(x), 1)
        # D_G_z = torch.sigmoid(self.dcdiscriminator(G_z))  # size (len(z), 1)

        D_x = self.dcdiscriminator(x) # size (len(x), 1)
        D_G_z = self.dcdiscriminator(G_z) # size (len(z), 1)
        

        # discriminator_loss = torch.mean(torch.sum(D_x, dim=1), dim=0) -     \
        #                             torch.mean(torch.sum(D_G_z, dim=1), dim = 0)
        discriminator_loss = -( D_x.mean() - D_G_z.mean() )                         # since we are doing gradient ascent on w parameterized for critic

        return discriminator_loss
    

    def forward(self, x):
        return self.dcgenerator(x)


    