import sys, os, bz2, random
import argparse
import pickle as cPickle
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import timm
import numpy as np
from num_sys_class import *
# from othermodels import resnet, vgg, cifar10_nn

'''
Environment Variables
'''
DATASETS = os.environ['ML_DATASETS']

'''
Helper functions to parse input
'''
batchsize_in = -1
dnn_in = ""
dataset_in = ""
format_in = ""
precision_in = ""
output_in = ""
cuda_in = True
injections_in = -1
injectionsLoc_in = 0
radix_in = -1
bitwidth_in = 32
bias_in = None
workers_in = -1
training_in = False
quantize_in = False
symmetricRanges_in = False
# singlebitflip_in = False
verbose_in = False
debug_in = False


# QUANTIZE_BITS = 8

def check_args(args=None):
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('-b', '--batchsize',
                        help='Batch Size for inference',
                        type=int,
                        required='True',
                        default=8)

    parser.add_argument('-n', '--dnn',
                        help='Neural network',
                        required='True',
                        default='alexnet')

    parser.add_argument('-d', '--dataset',
                        help='CIFAR10, CIFAR100, or IMAGENET',
                        required='True',
                        default='IMAGENET')

    parser.add_argument('-f', '--format',
                        help='Data format: fp32, fp16, bfloat16, fixedpt',
                        required='True',
                        default='fp32')

    parser.add_argument('-P', '--precision',
                        help='FP32 or FP16',
                        default='FP16')

    parser.add_argument('-o', '--output',
                        help='output path',
                        default='../output/')

    parser.add_argument('-C', '--cuda',
                        help='True or False',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)

    parser.add_argument('-i', '--injections',
                        help='The number of injections to perform',
                        type=int,
                        default=-1)

    parser.add_argument('-I', '--injectionLocation',
                        help='Injection Location (0-5)',
                        type=int,
                        default=-0)

    parser.add_argument('-R', '--radix',
                        help='Radix point for number format, from LSB',
                        type=int,
                        default=-1)

    parser.add_argument('-B', '--bitwidth',
                        help='Bitwidth of number format',
                        type=int,
                        default=32)

    parser.add_argument('-a', '--bias',
                        help='Bias value for AdaptivFloat number format',
                        type=int,
                        default=None)

    parser.add_argument('-r', '--training',
                        help='When enabled, this is training data. When disabled, this is testing data',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    parser.add_argument('-s', '--symmetricRanges',
                        help = 'When set to True, symmetric Range Detectors are used. Else, asymmetric Range Detectors are used.',
                        type = str2bool,
                        nargs = '?',
                        const = True,
                        default = False)


    parser.add_argument('-w', '--workers',
                        help='Number of workers for dataloader',
                        type=int,
                        default=0)

    parser.add_argument('-q', '--quantize',
                        help='Enable neuron quantization (def=8 bits)',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    parser.add_argument('-e', '--errormodel',
                        help='Single bit flip error model during quantization. -q needs to be enabled too',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    parser.add_argument('-v', '--verbose',
                        help='True or False',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    parser.add_argument('-D', '--debug',
                        help='True or False',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    results = parser.parse_args(args)

    global batchsize_in, dnn_in, dataset_in, format_in, precision_in, output_in, cuda_in, \
        bitwidth_in, radix_in, bias_in, \
        injections_in, injectionsLoc_in, training_in, workers_in, quantize_in, symmetricRanges_in, \
        verbose_in, debug_in
    # global singlebitflip_in

    batchsize_in = results.batchsize
    dnn_in = results.dnn
    dataset_in = results.dataset
    format_in = results.format
    precision_in = results.precision
    output_in = results.output
    cuda_in = results.cuda
    injections_in = results.injections
    injectionsLoc_in = results.injectionLocation
    radix_in = results.radix
    bitwidth_in = results.bitwidth
    bias_in = results.bias
    training_in = results.training
    symmetricRanges_in = results.symmetricRanges
    workers_in = results.workers
    quantize_in = results.quantize
    # singlebitflip_in = results.errormodel
    verbose_in = results.verbose
    debug_in = results.debug

    # make sure single bit flip is enabled only if quantization is enabled
    # if singlebitflip_in:
    #     assert (quantize_in)


def getBatchsize(): return batchsize_in
def getDNN(): return dnn_in
def getDataset(): return dataset_in
def getFormat(): return format_in
def getPrecision(): return precision_in
def getOutputDir(): return output_in
def getCUDA_en(): return cuda_in
def getInjections():
    if injections_in != -1 and injectionsLoc_in == 0:
        print("Warning: No injection location. Please include \"-I\" flag with value.")
    return injections_in
def getInjectionsLocation(): return injectionsLoc_in
def getRadix(): return radix_in
def getBitwidth(): return bitwidth_in
def getBias(): return bias_in
def getTraining_en(): return training_in
def getWorkers(): return workers_in
def getQuantize_en(): return quantize_in
# def getQuantizeBits(): return QUANTIZE_BITS
# def getSingleBitFlip_en(): return singlebitflip_in
def getSymmetricRanges_en(): return symmetricRanges_in
def getVerbose(): return verbose_in
def getDebug(): return debug_in


def printArgs():
    print('BATCH SIZE: \t%d\nDNN: \t\t%s\nDATASET: \t%sFORMAT: \t%s\n' \
          'PRECISION: \t%s\nOUTPUT: \t%s\nUSE_CUDA: \t%s\n' \
          'BIT-WIDTH: \t%s\nRADIX: \t%s\nBIAS: \t%s\n' \
          'INJECTIONS: \t%s\nINJECTIONS LOCATION: \t%s\nTRAINING DATA: \t%s\nWORKERS: \t%d\n' \
          'VERBOSE: \t%s\nDEBUG: \t\t%s\n' \
          % (batchsize_in, dnn_in, dataset_in, format_in, precision_in, output_in, cuda_in, \
             bitwidth_in, radix_in, bias_in, \
             injections_in, injectionsLoc_in, training_in, workers_in, verbose_in, debug_in))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def getNumSysName(name):
#     if name == "fp32":
#         return num_fp32
#     elif name == "fp16":
#         return fp16
#     elif name == "bfloat16":
#         return bfloat16
#     elif name == "fixedpt":
#         return num_fixed_pt

# returns the number of classes for common datasets
def getNumClasses(dataset):
    if(dataset == 'CIFAR10'):
        return 10
    elif(dataset == 'CIFAR100'):
        return 100
    elif(dataset == 'IMAGENET'):
        return 1000


def getNetwork(networkName, DATASET):
    ####### IMAGENET #######
    FB_repo = 'facebookresearch/deit:main'
    if DATASET == 'IMAGENET':
        # Convolution Neural Networks
        if networkName == "alexnet": MODEL = models.alexnet(pretrained=True, progress=True)
        elif networkName == "vgg11": MODEL = models.vgg11(pretrained=True, progress=True)
        elif networkName == "vgg13": MODEL = models.vgg13(pretrained=True, progress=True)
        elif networkName == "vgg16": MODEL = models.vgg16(pretrained=True, progress=True)
        elif networkName == "vgg19": MODEL = models.vgg19(pretrained=True, progress=True)
        elif networkName == "vgg11_bn": MODEL = models.vgg11(pretrained=True, progress=True)
        elif networkName == "vgg13_bn": MODEL = models.vgg13(pretrained=True, progress=True)
        elif networkName == "vgg16_bn": MODEL = models.vgg16(pretrained=True, progress=True)
        elif networkName == "vgg19_bn": MODEL = models.vgg19(pretrained=True, progress=True)
        elif networkName == "resnet18": MODEL = models.resnet18(pretrained=True, progress=True)
        elif networkName == "resnet34": MODEL = models.resnet34(pretrained=True, progress=True)
        elif networkName == "resnet50": MODEL = models.resnet50(pretrained=True, progress=True)
        elif networkName == "resnet101": MODEL = models.resnet101(pretrained=True, progress=True)
        elif networkName == "resnet152": MODEL = models.resnet152(pretrained=True, progress=True)
        elif networkName == "squeezenet1_0": MODEL = models.squeezenet1_0(pretrained=True, progress=True)
        elif networkName == "squeezenet1_1": MODEL = models.squeezenet1_1(pretrained=True, progress=True)
        elif networkName == "densenet121": MODEL = models.densenet121(pretrained=True, progress=True)
        elif networkName == "densenet169": MODEL = models.densenet169(pretrained=True, progress=True)
        elif networkName == "densenet201": MODEL = models.densenet201(pretrained=True, progress=True)
        elif networkName == "densenet161": MODEL = models.densenet161(pretrained=True, progress=True)
        elif networkName == "inceptionv3": MODEL = models.inception_v3(pretrained=True, progress=True)
        elif networkName == "googlenet": MODEL = models.googlenet(pretrained=True, progress=True)
        elif networkName == "shufflenet": MODEL = models.shufflenet_v2_x1_0(pretrained=True, progress=True)
        elif networkName == "mobilenet": MODEL = models.mobilenet_v2(pretrained=True, progress=True)
        elif networkName == "resnext50_32x4d": MODEL = models.resnext50_32x4d(pretrained=True, progress=True)

        # transformers
        elif networkName == "vit_base": MODEL = timm.create_model("vit_base_patch16_224", pretrained=True)
        elif networkName == "deit_base": MODEL = torch.hub.load(FB_repo, 'deit_base_patch16_224', pretrained=True)
        elif networkName == "deit_tiny": MODEL = torch.hub.load(FB_repo, 'deit_tiny_patch16_224', pretrained=True)

        # Error
        else:
            sys.exit("Network does not exist")

    elif DATASET == 'CIFAR10' or DATASET == 'CIFAR100':
        if networkName == "resnet18":
            MODEL = resnet.resnet18(pretrained=True)
        elif networkName == "vgg19_bn":
            MODEL = vgg.vgg19_bn(pretrained=True)
        elif networkName == "cifar10_nn_baseline":
            MODEL = cifar10_nn.baseline(pretrained=True, output_size=getNumClasses(DATASET))
        elif networkName == "cifar10_nn_v1":
            MODEL = cifar10_nn.v1(pretrained=True, output_size=getNumClasses(DATASET))
        elif networkName == "cifar10_nn_v2":
            MODEL = cifar10_nn.v2(pretrained=True, output_size=getNumClasses(DATASET))

        # Error
        else:
            sys.exit("Network does not exist")

    # model upgrades
    if getCUDA_en():
        MODEL = MODEL.cuda()
    if getPrecision() == 'FP16':
        MODEL = MODEL.half()

    return MODEL

def load_dataset(DATASET, BATCH_SIZE, workers=0, training=False, shuffleIn=False, include_id=True):
    if DATASET == 'CIFAR10':
        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
                ]
            )

        if include_id:
            testset = IdCifar10(root='./data', train=training, download=True, transform=transform)
        else:
            testset = datasets.CIFAR10(root='./data', train=training, download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                shuffle=shuffleIn, num_workers=workers, pin_memory=True)
        dataiter = iter(test_loader)

    elif DATASET == 'CIFAR100':
        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
                 ]
                )

        if include_id:
            testset = IdCifar100(root='./data', train=training, download=True, transform=transform)
        else:
            testset = datasets.CIFAR100(root='./data', train=training, download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                shuffle=shuffleIn, num_workers=workers, pin_memory=True)
        dataiter = iter(test_loader)

    elif DATASET == 'IMAGENET':
        if training == False:
            valdir = os.path.join(DATASETS + '/imagenet/', 'val')
        else:
            valdir = os.path.join(DATASETS + '/imagenet/', 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        if include_id: images = IdImageFolder(valdir, transform=transform)
        else: images = datasets.ImageFolder(valdir, transform=transform)

        val_loader = torch.utils.data.DataLoader(images, batch_size=BATCH_SIZE,
                                                 shuffle=shuffleIn, num_workers=workers, pin_memory=True)
        dataiter = iter(val_loader)

    return dataiter


# total_data refers to the total size of the data_loader, for all images desired
def load_custom_dataset(NETWORK, DATASET, BATCH_SIZE, good_images, total_data,
                        workers = 0, random=True, replacement=True, single=False, singleIndex=0):
    if random:
        if replacement:
            if single == False:
                custom_sampler = get_custom_sampler(good_images, total_data)
            else:
                custom_sampler = get_custom_sampler_single(good_images, singleIndex, total_data)

        else:
            custom_sampler = get_custom_sampler_no_replacement(good_images, total_data)
    else:
        custom_sampler = get_custom_sampler_full(good_images)

    if DATASET == 'CIFAR10':
            transform = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
                    ]
            )

            testset = IdCifar10(root='./data', train=False,
                                               download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            sampler=custom_sampler, num_workers=workers, pin_memory=True)
            dataiter = iter(test_loader)

    if DATASET == 'CIFAR100':
            transform = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
                    ]
            )

            testset = IdCifar100(root='./data', train=False,
                                               download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            sampler=custom_sampler, num_workers=workers, pin_memory=True)
            dataiter = iter(test_loader)

    if DATASET == 'IMAGENET':

            valdir = os.path.join(DATASETS + '/imagenet/', 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            images = IdImageFolder(valdir, transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize,
                                    ]))
            val_loader = torch.utils.data.DataLoader(images, batch_size=BATCH_SIZE,
                    num_workers = workers, sampler=custom_sampler, pin_memory=True)
            dataiter = iter(val_loader)

    return dataiter

class IdCifar10(datasets.CIFAR10):
    def __init__(self, root, train=False,
            transform=None, target_transform=None,
            download=False):

        super(datasets.CIFAR10, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                    ' You can use download=True to download it')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        self.img_names = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = cPickle.load(f)
                else:
                    entry = cPickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                self.img_names.extend(entry['filenames'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def __getitem__(self, index):
        img, target, path = self.data[index], self.targets[index], self.img_names[index]
        #img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path, index

class IdCifar100(IdCifar10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
            ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
            ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
            'filename': 'meta',
            'key': 'fine_label_names',
            'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class IdImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        item = super(IdImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return item[0], item[1], path, index

class Custom_Sampler(torch.utils.data.Sampler):
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)

"""
Edit this to make the random selector
Input: list of good indices
Return: list of indices that will be used to load data
"""
def random_selector(indices, total):
    return random.choices(indices, k=total)

def single_selector(indices, index, total):
    single_index = [indices[index]]
    return random.choices(single_index, k=total)

def random_selector_no_replacement(indices, total):
    return random.sample(indices, k=total)

def get_custom_sampler(indices, total):
    # Use random sampling with replacement
    indices = random_selector(indices, total)

    # Create custom sampler
    sampler = Custom_Sampler(indices)

    return sampler

def get_custom_sampler_single(indices, index, total):
    # Use random sampling with replacement
    indices = single_selector(indices, index, total)

    # Create custom sampler
    sampler = Custom_Sampler(indices)

    return sampler

def get_custom_sampler_no_replacement(indices, total):
    # Use random sampling with replacement
    indices = random_selector_no_replacement(indices, total)

    # Create custom sampler
    sampler = Custom_Sampler(indices)

    return sampler

def get_custom_sampler_full(indices):
    # Create custom sampler
    sampler = Custom_Sampler(indices)

    return sampler


#################################################################
############## HELPER METHODS FOR IMG PROCESSING  ###############
#################################################################
def getMaxClass(tensor, dim=0):
    argmax = torch.argmax(tensor).item()
    conf = torch.max(tensor).item() * 100
    top2 = torch.topk(tensor, k=2, dim=0)[0]
    diff_top2 = (top2[0] - top2[1]) * 100
    return argmax, conf, diff_top2.item()

def getMaxClass_parallel(tensor_in, dim=0):
    tensor = F.softmax(tensor_in)
    argmax = torch.argmax(tensor).item()
    conf = torch.max(tensor).item() * 100
    top2 = torch.topk(tensor, k=2, dim=0)[0]
    diff_top2 = (top2[0] - top2[1]) * 100
    return argmax, conf, diff_top2.item()

# from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def diff_top2(data):
    return (data[0] - data[1]).item() * 100


#################################################################
##################### HELPER METHODS FOR I/O ####################
#################################################################
def save_data(path, file_name, data, compress = True):
    if not os.path.exists(path):
        os.makedirs(path)
    output = path + file_name + ".p"
    f = bz2.BZ2File(output + ".bz2","wb") if compress else open(fname,"wb")
    cPickle.dump(data, f)
    f.close()



def load_file(file_name, compress = True):
    f = bz2.BZ2File(file_name + '.p.bz2', "rb") if compress else open(file_name.strip('.p.bz2'),"rb")
    fileIn= cPickle.load(f)
    f.close()
    return fileIn

#################################################################
################### HELPER METHODS FOR NUMSYS ###################
#################################################################
def getNumSysName(name, bits=16, radix_up=5, radix_down=10, bias=None):
    # common number systems in PyTorch
    if name == "fp32":
        return num_fp32(), name
    if name == "INT":
        assert(getQuantize_en())
        return num_fp32(), name
    elif name == "fp16":
        return num_fp16(), name
    elif name == "bfloat16":
        return num_bfloat16(), name

    # generic number systems in PyTorch
    elif name == "fp_n":
        return num_float_n(exp_len=radix_up, mant_len=radix_down), name
    elif name == "fxp_n":
        return num_fixed_pt(int_len=radix_up, frac_len=radix_down), name
    elif name == "block_fp":
        return block_fp(bit_width=bits, exp_len=radix_up, mant_len=radix_down), name
    elif name == "adaptive_fp":
        return adaptive_float(bit_width=bits, exp_len=radix_up, mant_len=radix_down, exp_bias=bias), name

    else:
        sys.exit("Number format not supported")

