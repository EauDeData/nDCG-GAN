import torch
import matplotlib.pyplot as plt
from datautils import *
from models import *
import loss as ndcg_loss


logfile = open('logfile.txt', 'a+')


def train(model, d_loss, r_loss, ndcg_loss, optimizers, data,  sch, sch_d, log = logfile, bsize = 2, w = [0.33, 0.33, 0.33], device = 'cuda'):

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
        [o.step() for o in optimizers[1:]] # First is discriminator
        #sch.step(generator_loss)


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
        #sch_d.step(dloss)

        # Verbose variables for generator
        dloss_cumsum_discriminator += dloss.item()

    invN = 1/(n+1)
    return dloss_cumsum_generator, dloss_cumsum_discriminator, rloss_cumsum/n, ndcg_loss_cumsum, generator_cumsum

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
    return  dloss_cumsum_generator, dloss_cumsum_discriminator, rloss_cumsum/n, ndcg_loss_cumsum, generator_cumsum


if __name__ == '__main__':
    device = 'cuda'
    EPOCHES = 100000
    bs = 32

    test_data = Yearbook(YEARBOOK_BASE + '/test_F.txt')
    train_data = Yearbook(YEARBOOK_BASE + '/test_M.txt')
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = bs, drop_last = True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = bs, num_workers = 4, drop_last = True)

    model = nDCG_GAN().to(device)

    #generators = [model.generator, model.visual_encoder, model.context_encoder, model.ranking, model.classfier]
    #discriminator = [model.discriminator]

    lr_dis = 1 * 5e-5
    lr_gen = 1 * 5e-5
    lr_rec = 1 * 1e-5
    lr_con = lr_rec
    lr_cla = 1 * 1e-5

    dis_params = list(model.discriminator.parameters())
    gen_params = list(model.generator.parameters())
    rec_params = list(model.visual_encoder.parameters())
    con_params = list(model.context_encoder.parameters())
    cla_params = list(model.classfier.parameters())

    dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = torch.optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)
    con_opt = torch.optim.Adam([p for p in con_params if p.requires_grad], lr=lr_rec)
    cla_opt = torch.optim.Adam([p for p in cla_params if p.requires_grad], lr=lr_cla)

    optimizers = [dis_opt, gen_opt, rec_opt, rec_opt, con_opt, cla_opt]

    discriminative_loss = torch.nn.BCELoss()
    regression_loss = torch.nn.MSELoss()    
    ranking_loss = ndcg_loss.DGCLoss()
    scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_generator)
    scheduler_disc = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_discriminator)


    dloss_list = []
    dloss_list_generator = []
    rloss_list = []

    for i in range(EPOCHES):
        print(f"EPOCH - {i}")
        test_out = test(model, discriminative_loss, regression_loss, ranking_loss, optimizers, test_loader, bsize = bs, device=device)
        print("Test:", test_out )
        dloss_list.append(test_out[1] / 2)
        dloss_list_generator.append(test_out[0])
        rloss_list.append(test_out[2])
        
        plt.plot(dloss_list, label = 'Classifier (discriminator)')
        plt.plot(dloss_list_generator, label = 'generator')
        plt.legend()
        plt.savefig(f'/home/adria/Desktop/GAN-nDCG/nDCG-GAN/code/dloss.png')
        plt.clf()

        plt.plot(rloss_list)
        plt.savefig(f'/home/adria/Desktop/GAN-nDCG/nDCG-GAN/code/rloss.png')
        plt.clf()


        print("Train:", train(model, discriminative_loss, regression_loss, ranking_loss, optimizers, train_loader, scheduler, scheduler_disc, bsize = bs,  device = device))
    print(test(model, discriminative_loss, regression_loss, ranking_loss, optimizers, test_loader,bsize=bs, device=device))


