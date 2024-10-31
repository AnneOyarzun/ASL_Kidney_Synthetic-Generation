"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch

torch.cuda.empty_cache()

torch.cuda.memory_summary(device=None, abbreviated=False)

class TrainOptions_res6:
    def __init__(self):
        # Initialize variables with default values
        self.batch_size = 4
        self.learning_rate = 0.0002
        self.lr = 0.0002 #help='initial learning rate for adam'
        self.num_epochs = 5
        self.epoch_count = 1 #help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.n_epochs = 2 #help='number of epochs with the initial learning rate'
        self.n_epochs_decay = 3 #help='number of epochs to linearly decay learning rate to zero')
        self.use_gpu = True
        self.verbose = True
        self.gpu_ids = [0]
        self.dataroot = 'D:/RM_RENAL/CycleGAN/dataset/'
        self.checkpoints_dir = 'D:/RM_RENAL/CycleGAN/checkpoints/56modelXCAT/perceptual_loss/'
        self.dataset_mode = 'unaligned'# [unaligned | aligned | single | colorization]'
        self.phase = 'train' # 'train, val, test
        self.max_dataset_size = float("inf") # help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.direction = 'BtoA' # help='AtoB or BtoA'
        self.input_nc = 1 #help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.output_nc = 1 #help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.preprocess = 'none'
        self.no_flip = True
        self.serial_batches = False #help='if true, takes images in order to make batches, otherwise takes them randomly
        self.num_threads = 0 #help='# threads for loading data' #set to 0 to avoid error
        self.model = 'cycle_gan' #help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        self.isTrain = True
        self.name = 'myExp_bs4_lr0.0002_l10_10_10_PL_1_BtoA_NoAug_res6_resized256' #help='name of the experiment. It decides where to store samples and models') 
        self.ngf = 64 #help='# of gen filters in the last conv layer'
        self.netG = 'resnet_6blocks' #help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self.norm = 'instance' #help='instance normalization or batch normalization [instance | batch | none]')
        self.no_dropout = True #help='no dropout for the generator
        self.init_type = 'normal' #help='network initialization [normal | xavier | kaiming | orthogonal]'
        self.init_gain = 0.02 #help='scaling factor for normal, xavier and orthogonal.'
        self.ndf = 64 #help='# of discrim filters in the first conv layer'
        self.netD = 'basic' #help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        self.n_layers_D = 3 #help='only used if netD==n_layers'
        self.pool_size = 50 #help='the size of image buffer that stores previously generated images'
        self.gan_mode = 'lsgan' #help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        self.beta1 = 0.5 #help='momentum term of adam'
        self.lr_policy = 'linear' #help='learning rate policy. [linear | step | plateau | cosine]'
        self.continue_train = False #action='store_true', help='continue training: load the latest model')
        self.load_iter = 0 #help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        self.epoch = 'latest' #help='which epoch to load? set to latest to use latest cached model')
        self.display_id = 1 #help='window id of the web display'
        self.no_html = False #action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.display_winsize = 256 #help='display window size for both visdom and HTML')
        self.display_port = 8097 #help='visdom port of the web display')
        self.use_wandb = False #action='store_true', help='if specified, then init wandb logging')
        self.wandb_project_name = 'CycleGAN-and-pix2pix' #help='specify wandb project name')
        self.display_ncols = 4 #help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.display_server = "http://localhost" #help='visdom server of the web display')
        self.display_env = 'main' #help='visdom display environment name (default is "main")')
        self.print_freq = 10 #help='frequency of showing training results on console')
        self.lambda_A = 10.0 #help='weight for cycle loss (A -> B -> A)')
        self.lambda_B = 10.0 #help='weight for cycle loss (B -> A -> B)')
        self.lambda_identity = 1.0
        self.lambda_gradient = False # Loss function → gradient loss (Bauer)
        self.which_model_feat = 'resnet18' #help='selects model to use for feature network')
        self.lambda_feat_AfB = 1 #help=weight for perception loss between real A and fake B '
        self.lambda_feat_BfA = 1 #help='weight for perception loss between real B and fake A ')
        self.lambda_feat_fArecB = 1 #help='weight for perception loss between fake A and reconstructed B ')
        self.lambda_feat_fBrecA = 1 #help='weight for perception loss between fake B and reconstructed A ')
        self.lambda_feat_ArecA = 1 #help='weight for perception loss between real A and reconstructed A ')
        self.lambda_feat_BrecB = 1 #help='weight for perception loss between real B and reconstruced B ')

        self.display_freq = 400 #help='frequency of showing training results on screen')
        self.save_latest_freq = 5000 #help='frequency of saving the latest results')
        self.save_epoch_freq = 5 #help='frequency of saving checkpoints at the end of epochs')
        self.update_html_freq = 1000 #help='frequency of saving training results to html')
        self.save_by_iter = False #action='store_true', help='whether saves model by iteration')
        self.preprocess = 'resize'
        self.load_size = 256


    def parse(self):
        # Return the variables
        return self


    def parse(self):
        # Return the variables
        return self


def train(opt): 
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    print('Training ended')

if __name__ == '__main__':
    # Resnet6
    opt = TrainOptions_res6()   # get training options
    train(opt)



    