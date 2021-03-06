# An Evaluation of Self-Supervised Pre-Training for Skin-Lesion Analysis

Hello! Here you will find the code to reproduce the results for the paper ["An Evaluation of Self-Supervised Pre-Training for Skin-Lesion Analysis"](https://arxiv.org/abs/2106.09229) 

![Evaluated Pipelines](./images/pipeline.jpeg)

---

## Datasets

To download all data to reproduce our work please use the links in below. In all experiments, we used subsets from the ISIC 2019 challenge training data for training, validating, and testing. More details about each dataset can be found in our paper. 

- [ISIC 2019](https://challenge2019.isic-archive.com/data.html)
- [ISIC 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification)
- [derm7pt-clinical](https://github.com/jeremykawahara/derm7pt)
- [derm7pt-dermato](https://github.com/jeremykawahara/derm7pt)
- [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

--- 

## Preparing Environment and Data 

We used nvidia-docker for all experiments. We made available the ``Dockerfile`` describing our development environment. 

Now, download all data linked in the previous section and adjust both data folder and label paths. 

For ISIC2019, we recommend setting proper data path for  [finetuning_ssl.py](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/finetuning_ssl.py#L300), 
[isic_contrastive_finetuning.py](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/isic_contrastive_finetuner.py#L313), 
and [main_isic_supcon.py](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/SupContrast/main_isic_supcon.py#L168)

We use all other datasets only on test stage. Then, set the correct image and label paths for each dataset [here](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/test_external_datasets.py#L25-L36).

---

## Self-supervised Checkpoints 

We used each model's weights to initialize a ResNet-50 encoder for fine-tuning experiments using self-supervised pre-trained models. Below you will find a list of each model's checkpoints to download.

Model | Checkpoint link | Notes
------------ | ------------- | ------------- 
[SimCLR](https://arxiv.org/abs/2002.05709) | https://github.com/google-research/simclr#pre-trained-models-for-simclrv1 | Weights [converted](https://github.com/tonylins/simclr-converter) from Tensorflow to PyTorch 
[SwAV](https://arxiv.org/abs/2006.09882) | https://github.com/facebookresearch/swav#model-zoo | -
[BYOL](https://arxiv.org/abs/2006.07733) | https://github.com/deepmind/deepmind-research/tree/master/byol#pretraining | Weights [converted](https://github.com/chigur/byol-convert) from JAX to PyTorch
[MoCo](https://arxiv.org/abs/2003.04297) | https://github.com/facebookresearch/moco#models | MoCo V2 checkpoint trained for 800 epochs 
[InfoMIN](https://arxiv.org/abs/2005.10243) | https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/docs/MODEL_ZOO.md | -

Once you downloaded all the weights, you need to set the correct path for each method [here](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/utils/misc.py#L66-L89).


---

## Running 


### Fine-tuning Only

To run the experiments regarding self-supervised and supervised models, we need to run ``finetuning_ssl.py``.

Essentially, to run a standard fine-tuning procedure you just need to specify which method (``--method`` parameter) 
among the available options {simclr, byol, swav, moco, infomin, baseline} and the folder containing the train and validation splits for the ISIC2019 dataset. The splits used in our paper are in the ``datasplits`` folder. The ``baseline`` stands for supervised training on ImageNet. A minor example of how our code: 

```bash
  python3 finetuning_ssl.py --method simclr --lr lr --batch_size batch_size --splits_folder /ssl-skin-lesions/datasplits/isic2019/splits/
```

To check all the parameters available please take a look at [here](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/finetuning_ssl.py#L261-L277)


### Pre-training and Fine-tuning


In this pipeline we perform an additional contrastive pre-training - which can use the supervised on the self-supervised version  - before fine-tuning.
We give more details of how to execute the contrastive pre-training at the folder [SupContrast](https://github.com/VirtualSpaceman/ssl-skin-lesions/tree/main/SupContrast).

When the pre-training finishes, just execute the ``isic_contrastive_finetuning.py`` passing the pre-trained model 
checkpoint on parameter ``--ckpt_path``. Such file is based on ``finetuning_ssl.py``, but with minor changes. 
We removed the ``--method`` parameter and fixed the ``simclr`` data augmentations during the fine-tuning. Except for the ``--method`` parameter, all the remaining ones are the same as explained in [Fine-tuning section](#fine-tuning). 

--- 

### Testing the models

You can use the file ``test_external_datasets.py`` to run the test step with a trained model. For example, 

```bash
  python3 test_external_datasets.py --dataset ds --ckpt_path path
```

or 

```bash
  python3 test_external_datasets.py --dataset ds --ckpt_path path --fromcl
```

if the evaluated checkpoint went through a contrastive pre-training, either supervised or self-supervised. 


We use test-time augmentation and evaluate the AUC over 50 copies. The datasets available for the ``--dataset`` parameter 
are {atlas-dermato, atlas-clinical, isic20, pad-ufes-20}. As we evaluated 5 distinct test datasets, we created a bash script to ease the whole setup in ``run_test_external.sh``.

---

## Top-5 Best Experiments

As mentioned in our paper, we train the top-5 best models under full- and low-data regime. Below, we describe the parameters for the top-5 best models for each evaluated pipeline.

### Hyperoptimized supervised Baseline (SUP -> FT)

We also made available the script ``run_supervied_hypersearch_finetuning.sh`` to run the hyperparameter search in 
supervised baseline as mentioned in our paper.  We describe top-5 best hyperparameter combination in the table below. 

Learning Rate (LR) | LR Scheduler | Batch Size | Balanced Batches?
------------ | ------------- | ------------- | ------------- 
0.009| plateau | 128 | Yes
0.002| plateau | 32 | Yes
0.005| plateau | 128 | Yes
0.003| plateau | 128 | Yes
0.0001| cosine | 32 | Yes

### Self-supervised (SSL -> FT)
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


## Acknowledgments
- L. Chaves is partially funded by QuintoAndar, and CAPES. 
- A. Bissoto is partially funded by FAPESP 2019/19619-7. 
- E. Valle is funded by CNPq 315168/2020-0.
- S. Avila is partially funded by CNPq PQ-2 315231/2020-3, and 
FAPESP 2013/08293-7.
- A. Bissoto and S. Avila are also partially funded by Google LARA 2020.
- The RECOD lab is funded by grants from FAPESP, CAPES, and CNPq.

