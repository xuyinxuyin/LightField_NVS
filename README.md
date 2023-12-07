# SE(3) Equivariant Convolution and Transformer in Ray Space
PyTorch implementation of paper ["SE(3) Equivariant Convolution and Transformer in Ray Space"](https://arxiv.org/pdf/2212.14871.pdf)


## Environment
This work follows the architecture of the work: [IBRNet: Learning Multi-View Image-Based Rendering](http://arxiv.org/abs/2102.13090) and we use the same environment.
To create an anaconda environment:
```
conda env create -f environment.yml
conda activate lightfield
```
## Datasets
We use the same datasets as IBRNet, please follow the [instruction](https://github.com/googleinterns/IBRNet) to download the datasets. 

## Running
```
example of multiple GPUs:
python -m torch.distributed.launch --nproc_per_node=8 train.py --config configs/train.txt
```

```
example of a single GPU:
python train.py --config configs/train.txt
```