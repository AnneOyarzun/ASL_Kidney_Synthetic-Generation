"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images_as_nifti
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

class TestOptions:
    def __init__(self):
        self.results_dir = 'D:/RM_RENAL/CycleGAN/results/' #help='saves results here.
        self.aspect_ratio = 1.0 #help='aspect ratio of result images'
        self.phase = 'test' #help='train, val, test, etc'
        # Dropout and Batchnorm has different behavioir during training and test.
        self.eval = True #action='store_true', help='use eval mode during test time.')
        self.num_test = 3894 #default=50, help='how many test images to run')
        # rewrite devalue values
        # To avoid cropping, the load_size should be the same as crop_size
        self.isTrain = False
        self.dataset_mode = 'single' #'unaligned'
        self.model_suffix = '_A' # Testear con generador A 
        self.dataroot = 'D:/RM_RENAL/CycleGAN/dataset/testB/'
        self.max_dataset_size = float("inf") # help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.direction = 'BtoA' # help='AtoB or BtoA'
        self.input_nc = 1 #help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.output_nc = 1 #help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.preprocess = 'none'
        self.model = 'test' #help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        self.gpu_ids = [0]
        self.checkpoints_dir = 'D:/RM_RENAL/CycleGAN/checkpoints/'
        #self.checkpoints_dir = 'Z:/CycleGAN/checkpoints'
        self.name = '56modelXCAT/perceptual_loss/myExp_bs4_lr0.0002_l10_10_10_PL_1_BtoA_NoAug_unet256_resized256/' #help='name of the experiment. It decides where to store samples and models')
        self.ngf = 64 #help='# of gen filters in the last conv layer'
        self.netG = 'unet_256' #help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self.norm = 'instance' #help='instance normalization or batch normalization [instance | batch | none]')
        self.no_dropout = True #help='no dropout for the generator
        self.init_type = 'normal' #help='network initialization [normal | xavier | kaiming | orthogonal]'
        self.init_gain = 0.02 #help='scaling factor for normal, xavier and orthogonal.'
        self.load_iter = 0 #help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        self.epoch = 5 #'latest' #'latest' #help='which epoch to load? set to latest to use latest cached model')
        self.verbose = True
        self.use_wandb = False #action='store_true', help='if specified, then init wandb logging')
        self.display_winsize = 256 #help='display window size for both visdom and HTML')
        self.preprocess = 'resize'
        self.load_size = 256

    def parse(self):
            # Return the variables
            return self       


if __name__ == '__main__':
    opt = TestOptions()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only 'supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        print(img_path)
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images_as_nifti(webpage, visuals, img_path, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML


## Link for pretrained models
#http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/