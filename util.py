import sys
import argparse

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
