# Contrastive Pre-training

This code is mainly based on the original implementation of [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) 
with minor adaptations to fit best in our needs. The original source code can be found [here](https://github.com/HobbitLong/SupContrast).


# Running

To run the pre-training step, we made available some bash scripts to run all experiments regarding the 
full- and low-data training scenarios. Please, refer to our paper for more details.  

Remember to properly set the dataset folder for your data set [here](https://github.com/VirtualSpaceman/ssl-skin-lesions/blob/main/SupContrast/main_isic_supcon.py#L168) 
and adjust the label's csv path inside the bash scripts. Richer details about each parameter in ``main_isic_supcon.py`` 
is in the source code. 
