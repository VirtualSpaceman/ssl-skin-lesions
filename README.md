# ssl-skin-lesions


# Datasets

You can download the data from the links in below. In all our experiments, we used only ISIC 2019 for training. More details about each dataset can be found in our paper. 

- [ISIC 2019](https://www.kaggle.com/andrewmvd/isic-2019)
- [ISIC 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification)
- [derm7pt-clinical](https://github.com/jeremykawahara/derm7pt)
- [derm7pt-dermato](https://github.com/jeremykawahara/derm7pt)
- [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

# Self-supervised Checkpoints 

Model | Checkpoint link | Notes
------------ | ------------- | ------------- 
[SimCLR](https://arxiv.org/abs/2002.05709) | https://github.com/google-research/simclr#pre-trained-models-for-simclrv1 | Weights [converted](https://github.com/tonylins/simclr-converter) from Tensorflow to PyTorch 
[SwAV](https://arxiv.org/abs/2006.09882) | https://github.com/facebookresearch/swav#model-zoo | -
[BYOL](https://arxiv.org/abs/2006.07733) | https://github.com/deepmind/deepmind-research/tree/master/byol#pretraining | Weights [converted](https://github.com/chigur/byol-convert) from JAX to PyTorch
[MoCo](https://arxiv.org/abs/2003.04297) | https://github.com/facebookresearch/moco#models | MoCo V2 checkpoint trained for 800 epochs 
[InfoMIN](https://arxiv.org/abs/2005.10243) | https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/docs/MODEL_ZOO.md | -
