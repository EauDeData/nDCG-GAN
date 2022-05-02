import torch
import matplotlib.pyplot as plt
from datautils import *
from models import *
import loss as ndcg_loss


logfile = open('logfile.txt', 'a+')


def train(model, d_loss, r_loss, ndcg_loss, optimizers, data, log = logfile, bsize = 2, w = [0.33, 0.33, 0.33], device = 'cuda'):

    dloss_cumsum_generator = 0
    dloss_cumsum_discriminator = 0

    generator_cumsum = 0
    rloss_cumsum = 0
    ndcg_loss_cumsum = 0

    true = torch.zeros((bsize, 2)).to(device)
    true[:, 0] = 1
    false = torch.zeros((bsize, 2)).to(device)
    false[:, 1] = 1

    for n, (target, yt, context, yc) in enumerate(data):


        target = target.to(device)
        context = context.to(device)
        yt, yc = yt.float().view(-1).to(device), yc.float().view(-1).to(device)

        [optim.zero_grad() for optim in optimizers]
        _, fake_chance_generator, fake_chance_discriminator, true_chance, predicted_year_true, predicted_year_fake, ranking = model(context, target)


        # Generator loss

        dloss = d_loss(fake_chance_generator, true)
        rloss = (r_loss(predicted_year_true.view(-1), yt) + r_loss(predicted_year_fake.view(-1), yc)) *.5
        #ranking_loss = ndcg_loss(ranking, yc) 

        generator_loss = w[0] * dloss + w[1] * rloss #+ w[2] * ranking_loss
        generator_loss.backward()
        optimizers[1].step()


        #Verbose variables for generator 
        dloss_cumsum_generator += dloss.item()
        generator_cumsum += generator_loss.item()
        rloss_cumsum += rloss.item()
        #ndcg_loss_cumsum += ranking_loss.item()


        # Discriminator loss
        fake_chance_discriminator = model.discriminator(fake_chance_discriminator.detach())
        dloss = (d_loss(fake_chance_discriminator, false) + d_loss(true_chance, true)) * .5
        dloss.backward()
        optimizers[0].step()

        # Verbose variables for generator
        dloss_cumsum_discriminator += dloss.item()

    invN = 1/(n+1)
    return dloss_cumsum_generator, dloss_cumsum_discriminator, rloss_cumsum, ndcg_loss_cumsum, generator_cumsum

def test(model, d_loss, r_loss, ndcg_loss, optimizers, data, log = logfile, bsize = 2, w = [0.33, 0.33, 0.33], device = 'cuda'):

    dloss_cumsum_generator = 0
    dloss_cumsum_discriminator = 0

    generator_cumsum = 0
    rloss_cumsum = 0
    ndcg_loss_cumsum = 0


    true = torch.zeros((bsize, 2)).to(device)
    true[:, 0] = 1
    false = torch.zeros((bsize, 2)).to(device)
    false[:, 1] = 1
    with torch.no_grad():
        for n, (target, yt, context, yc) in enumerate(data):

            target = target.to(device)
            context = context.to(device)
            yt, yc = yt.float().view(-1).to(device), yc.float().view(-1).to(device)

            img, fake_chance_generator, fake_chance_discriminator, true_chance, predicted_year_true, predicted_year_fake, ranking = model(context, target)


            # Generator loss

            #print(fake_chance_generator.shape, true.shape, end = '\r')

            #print(true.shape, fake_chance_generator.shape)
            dloss = d_loss(fake_chance_generator, true)
            rloss = (r_loss(predicted_year_true.view(-1), yt) + r_loss(predicted_year_fake.view(-1), yc)) *.5
            #ranking_loss = ndcg_loss(ranking, yc) 

            generator_loss = w[0] * dloss + w[1] * rloss #+ w[2] * ranking_loss

            #Verbose variables for generator 
            dloss_cumsum_generator += dloss.item()
            generator_cumsum += generator_loss.item()
            rloss_cumsum += rloss.item()
            #ndcg_loss_cumsum += ranking_loss.item()


            # Discriminator loss
            fake_chance_discriminator = model.discriminator(fake_chance_discriminator.detach())
            dloss = (d_loss(fake_chance_discriminator, false) + d_loss(true_chance, true))

            # Verbose variables for generator
            dloss_cumsum_discriminator += dloss.item()

            plt.imshow(img[0].squeeze().cpu().numpy(), cmap = 'gray')
            plt.axis('off')
            plt.savefig(f'/home/adria/Desktop/GAN-nDCG/nDCG-GAN/code/imgs/{n}.png')
            plt.clf()

    invN = 1/(n+1)
    return  dloss_cumsum_generator, dloss_cumsum_discriminator, rloss_cumsum, ndcg_loss_cumsum, generator_cumsum


if __name__ == '__main__':
    device = 'cuda'
    EPOCHES = 100
    bs = 128

    test_data = Yearbook(YEARBOOK_BASE + '/test_F.txt')
    train_data = Yearbook(YEARBOOK_BASE + '/test_M.txt')
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = bs, drop_last = True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = bs, num_workers = 4, drop_last = True)

    model = nDCG_GAN().to(device)

    generators = [model.generator, model.visual_encoder, model.context_encoder, model.ranking, model.classfier]
    discriminator = [model.discriminator]

    all = []
    [all.extend(list(x.parameters() )) for x in generators]
    optimizer_generator = torch.optim.RMSprop(all, lr = 1e-5) # Actually, we want to train different each part of the model

    all = []
    [all.extend(list(x.parameters() )) for x in discriminator]
    optimizer_discriminator = torch.optim.RMSprop(all, lr = 1e-5) # Actually, we want to train different each part of the model

    optimizers = [optimizer_discriminator, optimizer_generator]

    discriminative_loss = torch.nn.BCELoss()
    regression_loss = torch.nn.MSELoss()    
    ranking_loss = ndcg_loss.DGCLoss()

    dloss_list = []
    dloss_list_generator = []
    rloss_list = []

    for i in range(EPOCHES):
        test_out = test(model, discriminative_loss, regression_loss, ranking_loss, optimizers, test_loader, bsize = bs, device=device)
        print("Test:", test_out )
        dloss_list.append(test_out[1])
        dloss_list_generator.append(test_out[0])
        rloss_list.append(test_out[2])
        
        plt.plot(dloss_list)
        plt.plot(dloss_list_generator)
        plt.savefig(f'/home/adria/Desktop/GAN-nDCG/nDCG-GAN/code/dloss.png')
        plt.clf()

        plt.plot(rloss_list)
        plt.savefig(f'/home/adria/Desktop/GAN-nDCG/nDCG-GAN/code/rloss.png')
        plt.clf()


        print("Train:", train(model, discriminative_loss, regression_loss, ranking_loss, optimizers, train_loader, bsize = bs,  device = device))
    print(test(model, discriminative_loss, regression_loss, ranking_loss, optimizers, test_loader,bsize=bs, device=device))

