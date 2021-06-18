# Data splits

Here, we describe the data splits we use in our paper. Below, we give a brief description of the dataset, along with the content inside each folder. 

- **derm7pt-clin**: Clinical and out-distribution dataset used only in test stage. The folder contains only a csv file describing the labels and images.
- **derm7pt-derm**: Dermoscopy and out-distribution dataset used only in test stage. The folder contains only a csv file describing the labels and images.
- **isic2019**: Dermoscopy and in-distribution dataset employed on both train, validation, and test. Contains multiple subfolders regarding the full- and low-dataset splits. The subfolder named ``splits`` contains the split we used for full-data training. The subfolders containing the splits for 1% and 10% of the original training data are named as ``splits_x_id``, where ``x`` the percentage of the original dataset and ``id`` is the split id (up to 5) sampled from the full-data. Note that both validation and test split shared across all data scenarios.
- **isic2020**: Dermoscopy and in-distribution dataset used only in test stage. The folder contains only a csv file describing the labels and images.
- **pad-ufes-20**: Clinical and out-distribution dataset used only in test stage. The folder contains only a csv file describing the labels and images.


To use the ``isic2019`` data splits in both pre-training and fine-tuning protocol, you need to adjust the parameter ``split_folder`` setting the folder path according to each split. 
