# ssl-skin-lesions



# Datasets

You can download the data from the links in below. In all our experiments, we used subsets from the ISIC 2019 challenge training data for training, validating, and testing. More details about each dataset can be found in our paper. 

- [ISIC 2019](https://challenge2019.isic-archive.com/data.html)
- [ISIC 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification)
- [derm7pt-clinical](https://github.com/jeremykawahara/derm7pt)
- [derm7pt-dermato](https://github.com/jeremykawahara/derm7pt)
- [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)



# Preparing Data and Environment




# Self-supervised Checkpoints 

Here, we list each self-supervised model's checkpoints we used in our experiments. We used each model's weights to initialize a ResNet-50 encoder for fine-tuning experiments using self-supervised pre-trained models. 

Model | Checkpoint link | Notes
------------ | ------------- | ------------- 
[SimCLR](https://arxiv.org/abs/2002.05709) | https://github.com/google-research/simclr#pre-trained-models-for-simclrv1 | Weights [converted](https://github.com/tonylins/simclr-converter) from Tensorflow to PyTorch 
[SwAV](https://arxiv.org/abs/2006.09882) | https://github.com/facebookresearch/swav#model-zoo | -
[BYOL](https://arxiv.org/abs/2006.07733) | https://github.com/deepmind/deepmind-research/tree/master/byol#pretraining | Weights [converted](https://github.com/chigur/byol-convert) from JAX to PyTorch
[MoCo](https://arxiv.org/abs/2003.04297) | https://github.com/facebookresearch/moco#models | MoCo V2 checkpoint trained for 800 epochs 
[InfoMIN](https://arxiv.org/abs/2005.10243) | https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/docs/MODEL_ZOO.md | -

Once you download all weights, you need to set the correct path for each method [here](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/utils/misc.py#L66-L89).

# Runnning 


### Fine-tuning

To run the experiments regarding self-supervised and supervised models, we need to run ``finetuning_ssl.py``.

Essentially, to run a standard fine-tuning procudure you just need to specify which method (``--method`` parameter) you want among the available options {simclr, byol, swav, moco, infomin, baseline}. ``baseline`` stands for supervised training on ImageNet. For example, 

``
  python3 finetuning_ssl --method simclr --lr lr --batch_size batch_size 
``

To change any parameter you can take a look at all available options [here](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/finetuning_ssl.py#L261-L277)


### Pre-training and Fune-tuning

Now, we need to perform an extra contrastive pre-training - which can use the supervised on the self-supervised version  - before fine-tuning step.

### Testing the models

# Top-5 Best Experiments

As mentioned in our paper, we train the top-5 best models under full- and low-data regime. Below, we describe the parameters for the top-5 best models for each evaluated pipeline. 

### Hyperoptimized supervised Baseline

Learning Rate (LR) | LR Scheduler | Batch Size | Balanced Batches?
------------ | ------------- | ------------- | ------------- 
0.009| plateau | 128 | Yes
0.002| plateau | 32 | Yes
0.005| plateau | 128 | Yes
0.003| plateau | 128 | Yes
0.0001| cosine | 32 | Yes

### Self-supervised
Method | Learning Rate (LR) 
------------ | ------------- 
SimCLR | 0.01
SwAV | 0.01
BYOL | 0.01
BYOL | 0.001
InfoMIN | 0.001

### Self-Supervised Pre-training (SSL -> UCL -> FT)
Temperature | Pre-training batch size | Pre-training epochs | Balanced Batches?
------------ | ------------- | ------------- | ------------- 
0.1 | 80 | 50 | No
0.1 | 512 | 200 | Yes
0.5 | 512 | 200 | No
0.1 | 80 | 50 | Yes
1.0 | 512 | 200 | No

### Supervised Pre-training (SSL -> SCL -> FT)
Temperature | Pre-training batch size | Pre-training epochs | Balanced Batches?
------------ | ------------- | ------------- | ------------- 
1.0 | 80 | 50 | Yes
1.0 | 80 | 200 | Yes
0.5 | 80 | 200 | Yes
0.5 | 80 | 200 | No
0.5 | 80 | 50 | No

