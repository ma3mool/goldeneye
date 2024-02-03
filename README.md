<h1 align="center">
    GoldenBox
</h1>

<p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

<p align="center">
  <a href="#background">Overview</a> •
  <a href="#Installing">Installing</a> •
  <a href="#code">Code</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
<!--   <a href="#citation">Citation</a> • -->
  <a href="#license">License</a>
</p>

## Overview

GoldenEye Object Detection is an extension of the original [GoldenEye](https://github.com//ma3mool/goldeneye) functional simulator with fault injection capabilities for common and emerging numerical formats. Previously the simulator was only supported image classification models. Now, we have extended our use case to object detection models as well.

### Installing

**Ubuntu-20.04 or later**
1. Clone the goldeneye repository.
```bash
git clone https://github.com/sajidahmed12/goldeneye-object-detection
```

2. Download ninja-build, which is needed for qtorch.
```bash
sudo apt install ninja-build
```

3. Install the other project dependencies from the requirements.txt file.
```bash
pip install -r requirements.txt
```

### Testing

```bash
pytest val/test_num_sys.py
```

## Code

### Structure
The ```scripts``` folder includes wrappers around the goldeneye-obj framework to simplify its use. The ```src``` folder contains all of the core components, such as number system implementation, error injection routines, dataloaders, etc. The ```val``` folder is used for unit testing the code. You can run it using pytest to check that the installation process was successful.



Example Outputs are saved in this Google Drive [link](https://drive.google.com/drive/folders/1nP0pavu3vprPc9EvuahkF9UsETDlwFEp?usp=sharing)

## Example Commands 

Pre-processing 
```
python preprocess.py -b 16 -n frcnn -d COCO -w 8 -P FP32 -f fp_n  -C 0 [MS-COCO FP32](MS-COCO FP32])
```

Profiling
```
python profiling.py -b 16 -n frcnn -d COCO -w 16 -P FP32 -f fp_n -B 32 -R 23
```

Split Data
```
python split_data.py -b 16 -n frcnn -d COCO -o -w 16 -P FP32 -f fp_n -B 32 -R 23
```

Error Injections
```
python injections.py -b 16 -n frcnn -d COCO -w 16 -P FP32 -i 102400 -I 1 -f fp_n -B 32 -R 23
```

Post-processing 
```
python postprocess.py -b 16 -n frcnn -d COCO -w 16 -P FP32 -i 102400 -I 1 -f fp_n -B 32 -R 23
```

## Acknowledgements

- This Repository was forked from [Goldeneye](https://github.com/ma3mool/goldeneye/) developed and maintained by [Sajid Ahmed](https://sajidahmed12.github.io) & [Dr. Abdulrahman Mahmoud](https://ma3mool.github.io/)

## License
<a href="LICENSE">MIT License</a>
