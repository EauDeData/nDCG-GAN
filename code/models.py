from turtle import forward
import torch
import torchvision
import torch.nn as nn
import numpy as np

OUT_BOTTLENECK = 1000
OUT_ENCODERS = 512

class BaseBottleneck(nn.Module):
    '''
    Basse model that forwards an image into an embedding space
        Used at: Encoders, Discriminator and Classifier
    '''
    def __init__(self) -> None:

        super(BaseBottleneck, self).__init__()
        self.conv = torchvision.models.resnet101(pretrained=1)
        self.upchannels = nn.Conv2d(1, 3, 1)
        self.a = nn.ReLU()

    def forward(self, x):
        x = self.upchannels(x)
        x = self.conv(x)
        return x

class VisualEncoder(BaseBottleneck):
    def __init__(self) -> None:
        super(VisualEncoder, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, OUT_ENCODERS)

    def forward(self, x):
        x = self.a(super(VisualEncoder, self).forward(x))
        return self.out(x)
    

class ContextEncoder(BaseBottleneck):
    def __init__(self) -> None:
        super(ContextEncoder, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, OUT_ENCODERS)

    def forward(self, x):
        x = self.a(super(ContextEncoder, self).forward(x))
        return self.out(x)

class Discriminator(BaseBottleneck):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, 2)
        self.a_ = nn.Softmax()

    def forward(self, x):
        x = self.a(super(Discriminator, self).forward(x))
        return self.a_(self.out(x))

class Classifier(BaseBottleneck):

    def __init__(self) -> None:
        super(Classifier, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, 1)

    def forward(self, x):
        x = self.a(super(Classifier, self).forward(x))
        return self.out(x) * 10 + 1930 # TODO: Revisitar això

class RankingEncoder(BaseBottleneck):
    def __init__(self) -> None:
        super(RankingEncoder, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, OUT_ENCODERS)

    def forward(self, x):
        x = self.a(super(RankingEncoder, self).forward(x))
        return (self.out(x) * 10 + 1930 ).float() # TODO: Revisitar això

class Generator(nn.Module):

    # TODO: Incorporate Alpha and Betta

    def __init__(self) -> None:
        super(Generator, self).__init__()
        ngf = 3
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(OUT_ENCODERS * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Upsample(scale_factor = 2),

            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, 1, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1, 1, 1)
        return self.main(x)


class nDCG_GAN(nn.Module):

    def __init__(self) -> None:
        super(nDCG_GAN, self).__init__()
        self.visual_encoder, self.context_encoder, self.generator = VisualEncoder(), ContextEncoder(), Generator()
        self.discriminator, self.classfier, self.ranking = Discriminator(), Classifier(), RankingEncoder()
    
    def to(self, device):
        self.visual_encoder.to(device), self.context_encoder.to(device), self.generator.to(device), self.discriminator.to(device), self.classfier.to(device), self.ranking.to(device)
        return self

    def forward(self, x, t):

        context = torch.mean(torch.stack([self.context_encoder(x[i]) for i in range(x.shape[0])]), 1)
        visual = self.visual_encoder(t)
        input_generator = torch.concat((context, visual), 1)
        img_generated = self.generator(input_generator)

        fake_chance_gGradient = self.discriminator(img_generated)
        true_chance = self.discriminator(t)


        predicted_year_true = self.classfier(t)
        predicted_year_fake = self.classfier(img_generated)

        ranking = self.ranking(img_generated)

        return img_generated, fake_chance_gGradient, img_generated, true_chance, predicted_year_true, predicted_year_fake, ranking
        
if __name__ == '__main__':

    from datautils import *
    data = Yearbook(YEARBOOK_BASE + '/test_F.txt')
    model = nDCG_GAN()
    loader = torch.utils.data.DataLoader(data, batch_size = 1)


    for x, y, z, yz in loader:

        print(x.shape, z.shape)

        print(model(z, x))

        break
