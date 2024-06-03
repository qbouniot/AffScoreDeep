# From Alexnet to Transformers: Measuring the Non-linearity of Deep Neural Networks with Affine Optimal Transport

## Installation 

You can install all the required package using `conda`:
```
conda env create -f environment.yml
```

Or directly with `pip`:
```
pip install -r requirements.txt
```

## Compute non-linearity signature

To run the code and compute non-linearity signature of a given architecture `ARCH` on a given dataset `DATASET`:

```
python src/aff_scores_torch.py --model_name ARCH --val_dataset DATASET
```

For instance, to measure non-linearity signature of `alexnet` on `cifar10`:

```
python src/aff_scores_torch.py --model_name alexnet --val_dataset cifar10
```

`ARCH` should be the name of an architecture with pretrained model available on `torchvision`. You can find the list of all architectures [here](https://pytorch.org/vision/main/models.html). `DATASET` should be one among `cifar10, cifar100, imagenet, random, fashionMNIST`.
