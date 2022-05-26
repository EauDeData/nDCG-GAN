from turtle import forward
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import modules_tro

OUT_BOTTLENECK = 512
OUT_ENCODERS = 512

##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size=OUT_BOTTLENECK, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        model = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            model.append(nn.BatchNorm2d(out_size, 0.8))
        model.append(nn.LeakyReLU(0.2))
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        model = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        return out


class BaseBottleneck(nn.Module):
    '''
    Basse model that forwards an image into an embedding space
        Used at: Encoders, Discriminator and Classifier
    '''
    def __init__(self) -> None:

        super(BaseBottleneck, self).__init__()
        #torchvision.models.resnet101(pretrained=1)
        self.upchannels = nn.Conv2d(1, 3, 1)
        self.conv = modules_tro.ImageEncoder()
        self.a = nn.ReLU()

    def forward(self, x):
        x = self.upchannels(x)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        return x

class VisualEncoder(BaseBottleneck):
    def __init__(self) -> None:
        super(VisualEncoder, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, OUT_ENCODERS)

    def forward(self, x):
        x = super(VisualEncoder, self).forward(x)
        return self.out(x)
    

class ContextEncoder(BaseBottleneck):
    def __init__(self) -> None:
        super(ContextEncoder, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, OUT_ENCODERS)

    def forward(self, x):
        x = super(ContextEncoder, self).forward(x)
        return self.out(x)

class Discriminator(BaseBottleneck):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, 2)
        self.a_ = nn.Softmax()

    def forward(self, x):
        x = super(Discriminator, self).forward(x)
        return self.a_(self.out(x))

class Classifier(BaseBottleneck):

    def __init__(self) -> None:
        super(Classifier, self).__init__()
        self.out = nn.Linear(OUT_BOTTLENECK, 1)
        self.a = nn.ReLU()

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
        self.linear = nn.Linear(OUT_ENCODERS * 2, OUT_ENCODERS * 8 * 8)
        self.main = modules_tro.Decoder( dim=OUT_ENCODERS )

    def forward(self, x):
        x = F.relu(self.linear(x))
        return self.main(x.view(x.shape[0], -1, 8, 8))


class nDCG_GAN(nn.Module):

    def __init__(self) -> None:
        super(nDCG_GAN, self).__init__()
        self.visual_encoder, self.context_encoder, self.generator = VisualEncoder(), ContextEncoder(), Generator()
        self.discriminator, self.classfier, self.ranking = Discriminator(), Classifier(), RankingEncoder()
        self.device = 'cpu'
    
    def to(self, device):
        self.device = device
        self.visual_encoder.to(device), self.context_encoder.to(device), self.generator.to(device), self.discriminator.to(device), self.classfier.to(device), self.ranking.to(device)
        return self

    def forward(self, x, t):

        context = torch.mean(torch.stack([self.context_encoder(x[i]) for i in range(x.shape[0])]), 1).squeeze()
        visual = self.visual_encoder(t)

        input_generator = torch.cat((context, visual), -1)

        img_generated = self.generator(input_generator)

        fake_chance_gGradient = self.discriminator(img_generated)
        true_chance = self.discriminator(t)


        predicted_year_true = self.classfier(t)
        predicted_year_fake = self.classfier(img_generated)

        #ranking = self.ranking(img_generated)

        return img_generated, fake_chance_gGradient, img_generated, true_chance, predicted_year_true, predicted_year_fake, None
        
if __name__ == '__main__':



    from datautils import *
    data = Yearbook(YEARBOOK_BASE + '/test_F.txt')
    model = nDCG_GAN()
    loader = torch.utils.data.DataLoader(data, batch_size = 2)


    for x, y, z, yz in loader:

        print(x.shape, z.shape)

        print(model(z, x)[0].shape)

        break
