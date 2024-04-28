import torch
from torch import nn
import torch.nn.functional as F


# noise matrix (z) -> neural network g_theta(z) -> generate image (x)
class Generator(nn.Module):

    def __init__(self, z_dim_ip, img_dim_op) -> None:
        super().__init__()
        self.generator = nn.Sequential(

            nn.Linear(z_dim_ip, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, img_dim_op),
            # F.softplus()
            nn.Sigmoid()
            
        )

    def forward(self, input_x):
        return self.generator(input_x)
    

class DCGenerator(nn.Module):

    def __init__(self, z_ip_dim, img_dim) -> None:
        super(DCGenerator, self).__init__()

        self.dcgenerator = nn.Sequential(

            nn.Conv2d(in_channels=z_ip_dim, out_channels= 128, kernel_size=3, stride= 1, padding=1),
            nn.MaxPool2d(kernel_size= (2,2), stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels= 256, kernel_size=3, stride= 1, padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride= 2),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU()
          

        )

        self.classifier = nn.Sequential(

            nn.Linear(256*7*7, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, img_dim),
            nn.Sigmoid()


        )

    def forward(self, x):
        x = self.dcgenerator(x)
        x = torch.flatten(x, 1)  # Flatten the output of the convolutional layers
        x = self.classifier(x)
        return x                # shape should be equal to [batch_size, 28*28]


        



