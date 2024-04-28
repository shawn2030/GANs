import torch
import torch.nn as nn
import torch.nn.functional as F


# generated image g_theta(x) or input image (x) - > neural network -> real or fake classification 
class Discriminator(nn.Module):

    def __init__(self, img_dim_ip) -> None:
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(img_dim_ip, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),          # log probability
            # nn.Sigmoid()
        )


    def forward(self, input_img_x):
        return self.discriminator(input_img_x)


class DCDiscriminator(nn.Module):
    def __init__(self, input_img_x_dim) -> None:
        super(DCDiscriminator, self).__init__()

        self.dcdiscriminator = nn.Sequential(

            nn.Conv2d(in_channels=input_img_x_dim, out_channels= 128, kernel_size=3, stride= 1, padding=1),
            nn.MaxPool2d(kernel_size= (2,2), stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels= 256, kernel_size=3, stride= 1, padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride= 2),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),

        )

        self.classifier = nn.Sequential(

            nn.Linear(256*7*7, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 1)


        )

        # Only clip 'weights' part of Conv2D and Linear. Don't clip their 'bias' params. AND don't clip
        # any BatchNorm params
        self.params_to_clip = [
            layer.weight for layer in self.dcdiscriminator.children() if isinstance(layer, nn.Conv2d)
        ] + [
            layer.weight for layer in self.classifier.children() if isinstance(layer, nn.Linear)
        ]


    def weight_clip(self, c:float):
        with torch.no_grad():
            for param in self.params_to_clip:
                param.clip_(max=c,min= -c)
    

    def forward(self, x):
        x = self.dcdiscriminator(x)
        x = torch.flatten(x, 1)  # Flatten the output of the convolutional layers
        x = self.classifier(x)   # shape should be equal to [batch_size, output of last neuron 
                                 #                                  which is a regression task]
                                 
        return x


