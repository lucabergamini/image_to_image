import torch
import numpy
import argparse
from matplotlib import pyplot
numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)
from networks.cyclegan import CycleGan
from torch.autograd import Variable
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import progressbar
from torchvision.utils import make_grid
from generator.generator import DATASET
from scripts.utils import RollingMeasure
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CYCLEGAN")
    parser.add_argument("--train_folder_X",action="store",dest="train_folder_X")
    parser.add_argument("--test_folder_X",action="store",dest="test_folder_X")
    parser.add_argument("--train_folder_Y",action="store",dest="train_folder_Y")
    parser.add_argument("--test_folder_Y",action="store",dest="test_folder_Y")
    parser.add_argument("--n_epochs",default=12,action="store",type=int,dest="n_epochs")
    parser.add_argument("--lambda_mse",default=1,action="store",type=float,dest="lambda_mse")
    parser.add_argument("--lr",default=2e-04,action="store",type=float,dest="lr")
    parser.add_argument("--decay_lr",default=1,action="store",type=float,dest="decay_lr")

    args = parser.parse_args()

    train_folder_X = args.train_folder_X
    test_folder_X = args.test_folder_X
    train_folder_Y = args.train_folder_Y
    test_folder_Y = args.test_folder_Y

    n_epochs = args.n_epochs
    lambda_mse = args.lambda_mse
    lr = args.lr
    decay_lr = args.decay_lr
    net = CycleGan(initial_fetures_map_enocer=64).cuda()
    print(net)
    #exit()
    writer = SummaryWriter(comment="_CYCLE")

    # DATASET
    dataloader_X = torch.utils.data.DataLoader(DATASET(train_folder_X), batch_size=1,
                                               shuffle=False, num_workers=2)
    dataloader_Y = torch.utils.data.DataLoader(DATASET(train_folder_Y), batch_size=1,
                                               shuffle=False, num_workers=2)
    # DATASET for test
    # if you want to split train from test just move some files in another dir
    dataloader_test_X = torch.utils.data.DataLoader(DATASET(test_folder_X), batch_size=100,
                                                    shuffle=False, num_workers=1)
    dataloader_test_Y = torch.utils.data.DataLoader(DATASET(test_folder_Y), batch_size=100,
                                                    shuffle=False, num_workers=1)


    # OPTIM-LOSS
    # an optimizer for each of the sub-networks, so we can selectively backprop
    optimizer_autoencoder = Adam(params=[p for p in net.autoencoder_G.parameters()]+[p for p in net.autoencoder_F.parameters()],lr=lr,betas=(0.5,0.999))
    lr_autoencoder = ExponentialLR(optimizer_autoencoder, gamma=decay_lr)
    optimizer_discriminator = Adam(params=[p for p in net.patch_G.parameters()]+[p for p in net.patch_F.parameters()],lr=lr,betas=(0.5,0.999))
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=decay_lr)

    batch_number = len(dataloader_X)
    step_index = 0
    widgets = [

        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('loss_cycle_G'),
        ' ',
        progressbar.DynamicMessage('loss_cycle_F'),
        ' ',
        progressbar.DynamicMessage('loss_autoencoder_G'),
        ' ',
        progressbar.DynamicMessage('loss_autoencoder_F'),
        ' ',
        progressbar.DynamicMessage('loss_discriminator_G'),
        ' ',
        progressbar.DynamicMessage('loss_discriminator_F'),
        ' ',
        progressbar.DynamicMessage("epoch")
    ]
    # for each epoch
    for i in range(n_epochs):
        progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0,
                                           widgets=widgets).start()
        # reset rolling average
        loss_cycle_G_mean = RollingMeasure()
        loss_cycle_F_mean = RollingMeasure()

        loss_autoencoder_G_mean = RollingMeasure()
        loss_autoencoder_F_mean = RollingMeasure()
        loss_discriminator_G_mean = RollingMeasure()
        loss_discriminator_F_mean = RollingMeasure()
        print("LR:{}".format(lr_autoencoder.get_lr()))

        # for each batch
        for j, (data_X,data_Y) in enumerate(zip(dataloader_X, dataloader_Y)):
            a = time.time()
            # set to train mode
            train_batch_size = len(data_X)
            net.train()
            # target and input are the same images
            data_X = Variable(data_X, requires_grad=False).float().cuda()
            data_Y = Variable(data_Y, requires_grad=False).float().cuda()

            # get output
            ten_original_X, ten_original_Y, ten_classifications_G, ten_classifications_F, ten_X_Y_X, ten_Y_X_Y = net(data_X,data_Y)
            print(time.time()-a)
            # loss, nothing special here
            l1_G, gan_autoencoder_G, gan_discriminator_G, l1_F, gan_autoencoder_F, gan_discriminator_F = CycleGan.loss(ten_original_X, ten_original_Y, ten_classifications_G, ten_classifications_F, ten_X_Y_X, ten_Y_X_Y,train_batch_size)
            # THIS IS THE MOST IMPORTANT PART OF THE CODE
            print(time.time()-a)
            loss_autoencoder_G = lambda_mse*l1_G + gan_autoencoder_G
            loss_autoencoder_F = lambda_mse*l1_F + gan_autoencoder_F
            loss_discriminator_G = gan_discriminator_G
            loss_discriminator_F = gan_discriminator_F

            # register mean values of the losses for logging
            loss_cycle_G_mean(l1_G.data.cpu().numpy()[0])
            loss_cycle_F_mean(l1_F.data.cpu().numpy()[0])
            loss_autoencoder_G_mean(loss_autoencoder_G.data.cpu().numpy()[0])
            loss_autoencoder_F_mean(loss_autoencoder_F.data.cpu().numpy()[0])
            loss_discriminator_G_mean(loss_discriminator_G.data.cpu().numpy()[0])
            loss_discriminator_F_mean(loss_discriminator_F.data.cpu().numpy()[0])

            print(time.time()-a)

            # BACKPROP
            # clean grads
            net.zero_grad()
            loss_autoencoder = loss_autoencoder_F+loss_autoencoder_G
            loss_autoencoder.backward(retain_graph=True)
            optimizer_autoencoder.step()
            #clean the discriminator
            net.patch_F.zero_grad()
            loss_discriminator_F.backward(retain_graph=False)
            net.patch_G.zero_grad()
            loss_discriminator_G.backward(retain_graph=False)
            optimizer_discriminator.step()
            print(time.time()-a)
            # LOGGING
            progress.update(progress.value + 1, loss_cycle_G=loss_cycle_G_mean.measure,
                            loss_cycle_F=loss_cycle_F_mean.measure,
                           loss_autoencoder_G=loss_autoencoder_G_mean.measure,
                           loss_autoencoder_F=loss_autoencoder_F_mean.measure,
                           loss_discriminator_G=loss_discriminator_G_mean.measure,
                           loss_discriminator_F=loss_discriminator_F_mean.measure,
                           epoch=i + 1)


            if j== 8:
                exit()
        # EPOCH END
        lr_autoencoder.step()
        lr_discriminator.step()
        progress.finish()

        writer.add_scalar('loss_autoencoder_G', loss_autoencoder_G_mean.measure, step_index)
        writer.add_scalar('loss_autoencoder_F', loss_autoencoder_F_mean.measure, step_index)
        writer.add_scalar('loss_cycle_G', loss_cycle_G_mean.measure, step_index)
        writer.add_scalar('loss_cycle_F', loss_cycle_F_mean.measure, step_index)
        writer.add_scalar('loss_discriminator_G', loss_discriminator_G_mean.measure, step_index)
        writer.add_scalar('loss_discriminator_F', loss_discriminator_F_mean.measure, step_index)


        # X_Y_X
        ten_X_Y_X = ten_X_Y_X.data.cpu()
        # porto in range 0-1
        ten_X_Y_X = (ten_X_Y_X + 1) / 2.0
        out = make_grid(ten_X_Y_X, nrow=8)
        writer.add_image("X_Y_X", out, step_index)
        # Y_X_Y
        ten_Y_X_Y = ten_Y_X_Y.data.cpu()
        # porto in range 0-1
        ten_Y_X_Y = (ten_Y_X_Y + 1) / 2.0
        out = make_grid(ten_Y_X_Y, nrow=8)
        writer.add_image("Y_X_Y", out, step_index)
        # original
        ten_original_X = ten_original_X.data.cpu()
        # porto in range 0-1
        ten_original_X = (ten_original_X + 1) / 2.0
        out = make_grid(ten_original_X, nrow=8)
        writer.add_image("X", out, step_index)
        ten_original_Y = ten_original_Y.data.cpu()
        # porto in range 0-1
        ten_original_Y = (ten_original_Y + 1) / 2.0
        out = make_grid(ten_original_Y, nrow=8)
        writer.add_image("Y", out, step_index)

        # for j, (data_X,data_Y) in enumerate(zip(dataloader_test_X,dataloader_test_Y)):
        #     net.eval()
        #
        #     data_X = Variable(data_X, volatile=True).float().cuda()
        #     data_Y = Variable(data_Y, volatile=True).float().cuda()
        #     ten_original_X, ten_original_Y, ten_X_Y, ten_Y_X, ten_X_Y_X, ten_Y_X_Y= net(data_X,data_Y)
        #     #X_Y
        #     ten_X_Y = ten_X_Y.data.cpu()
        #     #porto in range 0-1
        #     ten_X_Y = (ten_X_Y+1)/2.0
        #     out = make_grid(ten_X_Y, nrow=8)
        #     writer.add_image("X_Y", out, step_index)
        #     #Y_X
        #     ten_Y_X = ten_Y_X.data.cpu()
        #     #porto in range 0-1
        #     ten_Y_X = (ten_Y_X+1)/2.0
        #     out = make_grid(ten_Y_X, nrow=8)
        #     writer.add_image("Y_X", out, step_index)
        #     # X_Y_X
        #     ten_X_Y_X = ten_X_Y_X.data.cpu()
        #     #porto in range 0-1
        #     ten_X_Y_X = (ten_X_Y_X+1)/2.0
        #     out = make_grid(ten_X_Y_X, nrow=8)
        #     writer.add_image("X_Y_X", out, step_index)
        #     #Y_X_Y
        #     ten_Y_X_Y = ten_Y_X_Y.data.cpu()
        #     #porto in range 0-1
        #     ten_Y_X_Y = (ten_Y_X_Y+1)/2.0
        #     out = make_grid(ten_Y_X_Y, nrow=8)
        #     writer.add_image("Y_X_Y", out, step_index)
        #     #original
        #     ten_original_X = ten_original_X.data.cpu()
        #     #porto in range 0-1
        #     ten_original_X = (ten_original_X+1)/2.0
        #     out = make_grid(ten_original_X, nrow=8)
        #     writer.add_image("X", out, step_index)
        #     ten_original_Y = ten_original_Y.data.cpu()
        #     #porto in range 0-1
        #     ten_original_Y = (ten_original_Y+1)/2.0
        #     out = make_grid(ten_original_Y, nrow=8)
        #     writer.add_image("Y", out, step_index)
        #     break

        step_index += 1
    exit(0)
