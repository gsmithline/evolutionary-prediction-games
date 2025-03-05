# Evolutionary Prediction Games

Comments are welcome!

## Environment

Analysis and simulations:
```
conda create -n evoml anaconda conda-forge::cvxpy
```

Experiments with CIFAR-10/MNIST:
```
conda create -y -n ffcv python=3.9 anaconda pytorch torchvision torchaudio pytorch-cuda cupy pkg-config compilers libjpeg-turbo opencv cudatoolkit numba pyyaml -c pytorch -c nvidia -c conda-forge
conda activate ffcv
pip install ffcv
```
