# goldeneye

requires `sudo apt install ninja-build` for qtorch

requires `pip install torch torchvision qtorch`

To add custom models and/or pretrained weights:
1. add your model to `src/othermodels/`
2. add pt files to `src/othermodels/state_dicts`
3. modify `getNetwork()` in `src/util.py` to correspond to the dataset and model name. 
4. also add the appropriate `import` in `/src/util.py`

Your model.py file in `src/othermodels/` should have a few general attributes:
1. A codeblock similar to easily find your model versions 
```
__all__ = [
    "baseline",
    "v1",
    "v2",
]
```
2. A separate function for each name in Step 1, which instantiates your model and extracts the correct model parameters from `/src/othermodels/state_dicts/`
As an example, check out: https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py

