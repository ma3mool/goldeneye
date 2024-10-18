<h1 align="center">
    Goldeneye
</h1>

<p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

<p align="center">
  <a href="#background">Background</a> •
  <a href="#usage">Usage</a> •
  <a href="#code">Code</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a>
</p>

## Background

GoldenEye is a functional simulator with fault injection capabilities for common and emerging numerical formats, implemented for the PyTorch deep learning framework. GoldenEye provides a unified framework for numerical format evaluation of DNNs, including traditional number systems such as fixed and floating point, as well as recent DNN-inspired formats such as block floating point and AdaptivFloat. Additionally, GoldenEye enables single- and multi- bit flips at various logical and functional points during a value’s lifetime for resiliency analysis, including for the first time attention to numerical values’ hardware metadata. GoldenEye is an easy-to-use, extensible, versatile, and fast tool for dependability research and future DNN accelerator design.

![](https://user-images.githubusercontent.com/89948656/176387208-cfd64047-3841-4abf-bf54-5d2e63f5a2e5.png)


## Usage

Take a look at our documentation [here](https://goldeneyedocs.readthedocs.io/en/stable/index.html).
Check this [Colab notebook](https://colab.research.google.com/drive/1Om-Wg6wLOeaAKRWaDcpjoZZdF1LwXcCI) for a demo.

### Installing

**Ubuntu with Sudo Privileges**
1. Recursively clone the goldeneye repository.
```bash
git clone --recurse-submodules git@github.com:ma3mool/goldeneye.git
```

2. Download ninja-build which is needed for qtorch.
```bash
sudo apt install ninja-build
```

3. Download the other project dependencies. Please make sure you are inside the goldeneye folder when applying this command.
```bash
pip install -r requirements.txt
```

4. Setup environment variable (replace with the directory where the imagenet dataset is downloaded).
```bash
ML_DATASETS=/dir/to/imagenet/
```

**Docker**
1. Recursively clone the goldeneye repository.
```bash
git clone --recurse-submodules git@github.com:ma3mool/goldeneye.git
```

2. Pull the goldeneye docker image and rename it to simply the next steps
```bash
docker pull goldeneyetool/goldeneye:latest
docker image tag goldeneyetool/goldeneye goldeneye
```

3. Within the goldeneye folder, run the shell on the pulled docker image. Make sure to replace [/path/to/imagenet] with the actual path to your downloaded imagenet dataset.
```bash
cd goldeneye
docker run -ti 
    --mount type=bind,source=`pwd`/src/,target=/src 
    --mount type=bind,source=`pwd`/val/,target=/val 
    --mount type=bind,source=`pwd`/scripts/,target=/scripts 
    --mount type=bind,source=[/path/to/imagenet],target=/datasets/imagenet 
    goldeneye
```

### Testing

```bash
pytest val/test_num_sys.py
```

## Code

### Structure
The ```scripts``` folder includes wrappers around the goldeneye framework to simplify its use. The ```src``` folder contains all of the goldeneye core logic such as number system implementation and error injection routines. The ```val``` folder is used for unit-testing the code. You can run it using pytest to check that the installation process was successful.

## Acknowledgements

- Tarek Aloui (Harvard)
- David Brooks (Harvard)
- Abdulrahman Mahmoud (Harvard)
- Joshua Park (Harvard)
- Thierry Tambe (Harvard)
- Gu-Yeon Wei (Harvard)

## Citation

If you use or reference Goldeneye, please cite:

```
@INPROCEEDINGS{GoldeneyeMahmoudTambeDSN2022,
author={A. {Mahmoud} and T. {Tambe} and T. {Aloui} and D. {Brooks} and G. {Wei}},
booktitle={2022 52nd Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)},
title={GoldenEye:  A Platform for Evaluating Emerging Data Formats in DNN Accelerators},
year={2022},
}
```

## License

<a href="LICENSE">MIT License</a>
