import torch
import numpy
import argparse
from matplotlib import pyplot
numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)
from networks.cyclegan import AdversarialAutoEncoder
from torch.autograd import Variable
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import progressbar
from torchvision.utils import make_grid
from generator.generator import DATASET
from scripts.utils import RollingMeasure

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CYCLEGAN")
    parser.add_argument("--train_folder_X",action="store",dest="train_folder_X")
    parser.add_argument("--test_folder_X",action="store",dest="test_folder_X")
    parser.add_argument("--n_epochs",default=12,action="store",type=int,dest="n_epochs")
    parser.add_argument("--lambda_mse",default=1,action="store",type=float,dest="lambda_mse")
    parser.add_argument("--lr",default=2e-3,action="store",type=float,dest="lr")
    parser.add_argument("--decay_lr",default=1,action="store",type=float,dest="decay_lr")

    args = parser.parse_args()

    train_folder_X = args.train_folder_X
    test_folder_X = args.test_folder_X


    n_epochs = args.n_epochs
    lambda_mse = args.lambda_mse
    lr = args.lr
    decay_lr = args.decay_lr
    net = AdversarialAutoEncoder().cuda()
    #print(net)
    #exit()
    writer = SummaryWriter(comment="_ADV")

    # DATASET
    dataloader_X = torch.utils.data.DataLoader(DATASET(train_folder_X), batch_size=16,
                                               shuffle=True, num_workers=2)

    # DATASET for test
    # if you want to split train from test just move some files in another dir
    dataloader_test_X = torch.utils.data.DataLoader(DATASET(test_folder_X), batch_size=100,
                                                    shuffle=False, num_workers=1)



    # OPTIM-LOSS
    # an optimizer for each of the sub-networks, so we can selectively backprop
    optimizer_autoencoder = Adam(params=net.parameters(),lr=lr)
    lr_autoencoder = ExponentialLR(optimizer_autoencoder, gamma=decay_lr)

    batch_number = len(dataloader_X)
    step_index = 0
    widgets = [

        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('loss_autoencoder_G'),
        ' ',
        progressbar.DynamicMessage("epoch")
    ]
    # for each epoch
    for i in range(n_epochs):
        progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0,
                                           widgets=widgets).start()
        # reset rolling average

        loss_autoencoder_G_mean = RollingMeasure()

        #print("LR:{}".format(lr_encoder.get_lr()))

        # for each batch
        for j, data_X in enumerate(dataloader_X):
            # set to train mode
            train_batch_size = len(data_X)
            net.train()
            # target and input are the same images
            data_X = Variable(data_X, requires_grad=False).float().cuda()

            # get output
            ten_original_X, ten_reconstructed_X = net(data_X)
            # loss, nothing special here
            l1_G = AdversarialAutoEncoder.loss(ten_original_X,ten_reconstructed_X)
            # THIS IS THE MOST IMPORTANT PART OF THE CODE


            # register mean values of the losses for logging

            loss_autoencoder_G_mean(l1_G.data.cpu().numpy()[0])


            # BACKPROP
            # clean grads
            net.zero_grad()
            loss_autoencoder = l1_G
            loss_autoencoder.backward(retain_graph=True)
            optimizer_autoencoder.step()


            # LOGGING
            progress.update(progress.value + 1,
                           loss_autoencoder_G=loss_autoencoder_G_mean.measure,
                           epoch=i + 1)



        # EPOCH END
        lr_autoencoder.step()
        progress.finish()

        writer.add_scalar('loss_autoencoder_G', loss_autoencoder_G_mean.measure, step_index)



        for j, data_X in enumerate(dataloader_test_X):
            net.eval()

            data_X = Variable(data_X, volatile=True).float().cuda()
            ten_original_X, ten_reconstructed_X= net(data_X)
            #X_
            ten_reconstructed_X = ten_reconstructed_X.data.cpu()
            #porto in range 0-1
            ten_reconstructed_X = (ten_reconstructed_X+1)/2.0
            out = make_grid(ten_reconstructed_X, nrow=8)
            writer.add_image("X_Y", out, step_index)
            #original
            ten_original_X = ten_original_X.data.cpu()
            #porto in range 0-1
            ten_original_X = (ten_original_X+1)/2.0
            out = make_grid(ten_original_X, nrow=8)
            writer.add_image("X", out, step_index)

            break

        step_index += 1
    exit(0)
