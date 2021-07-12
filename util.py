import sys, os, bz2
import argparse
import pickle as cPickle
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import timm


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
precision_in = ""
output_in = ""
cuda_in = True
injections_in = -1
workers_in = -1
training_in = False
# quantize_in = False
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

    parser.add_argument('-r', '--training',
                        help='When enabled, this is training data. When disabled, this is testing data',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

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

    global batchsize_in, dnn_in, dataset_in, precision_in, output_in, cuda_in, \
        injections_in, training_in, workers_in, \
        verbose_in, debug_in
    # global quantize_in, singlebitflip_in

    batchsize_in = results.batchsize
    dnn_in = results.dnn
    dataset_in = results.dataset
    precision_in = results.precision
    output_in = results.output
    cuda_in = results.cuda
    injections_in = results.injections
    training_in = results.training
    workers_in = results.workers
    # quantize_in = results.quantize
    # singlebitflip_in = results.errormodel
    verbose_in = results.verbose
    debug_in = results.debug

    # make sure single bit flip is enabled only if quantization is enabled
    # if singlebitflip_in:
    #     assert (quantize_in)


def getBatchsize(): return batchsize_in
def getDNN(): return dnn_in
def getDataset(): return dataset_in
def getPrecision(): return precision_in
def getOutputDir(): return output_in
def getCUDA_en(): return cuda_in
def getInjections(): return injections_in
def getTraining_en(): return training_in
def getWorkers(): return workers_in
# def getQuantize_en(): return quantize_in
# def getQuantizeBits(): return QUANTIZE_BITS
# def getSingleBitFlip_en(): return singlebitflip_in
def getVerbose(): return verbose_in
def getDebug(): return debug_in


def printArgs():
    print('BATCH SIZE: \t%d\nDNN: \t\t%s\nDATASET: \t%s\n' \
          'PRECISION: \t%s\nOUTPUT: \t%s\nUSE_CUDA: \t%s\n' \
          'INJECTIONS: \t%s\nTRAINING DATA: \t%s\nWORKERS: \t%d\n' \
          'VERBOSE: \t%s\nDEBUG: \t\t%s\n' \
          % (batchsize_in, dnn_in, dataset_in, precision_in, output_in, cuda_in, \
             injections_in, training_in, workers_in, verbose_in, debug_in))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getNetwork(networkName, DATASET):
    ####### IMAGENET #######
    if DATASET == 'IMAGENET':
        FB_repo = 'facebookresearch/deit:main'
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

    # model upgrades
    if getPrecision() == 'FP16':
        MODEL = MODEL.half()
    if getCUDA_en():
        MODEL = MODEL.cuda()

    return MODEL

def load_dataset(DATASET, BATCH_SIZE, workers=0, training=False, shuffleIn=False):
    if DATASET == 'CIFAR10':
        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
                ]
            )
        testset = IdCifar10(root='./data', train=training,
                download=True, transform=transform)
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
        testset = IdCifar100(root='./data', train=training,
                download=True, transform=transform)
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
        images = IdImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        val_loader = torch.utils.data.DataLoader(images, batch_size=BATCH_SIZE, shuffle=shuffleIn, num_workers=workers, pin_memory=True)
        dataiter = iter(val_loader)

    return dataiter


class IdImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        item = super(IdImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return item[0], item[1], path, index


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
