from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from argparse import ArgumentParser

import os
import numpy as np

from utils import misc
from isic_contrastive_finetuner import ISICFineTuner
from finetuning_ssl import SSLModel
from custom_dataset import CSVDatasetWithName



if __name__ == "__main__":
    #Insert the datafolder path for each dataset
    IMG_PATHS = {'atlas-dermato': '/datasplits/atlas-rgb', 
                 'atlas-clinical': '/datasplits/atlas-clinical-rgb' ,
                 'isic20': '/datasplits/ISIC_2020_Training_Input',
                 'pad-ufes-20': '/datasplits/pad-ufes-20/'
    }
    
    #Insert the labels path for each dataset
    LABELS_PATHS = {'atlas-dermato': '/datasplits/derm7pt-derm/atlas-dermato-all.csv', 
                    'atlas-clinical': '/datasplits/derm7pt-clin/atlas-clinical-all.csv',
                    'isic20': '/datasplits/isic2020/isic2020-subset-test.csv',
                    'pad-ufes-20': '/datasplits/padufes20/pad-ufes-20-labels.csv'
    }
    
    METHODS = ['simclr', 'baseline', 'swav', 'infomin', 'byol', 'moco']
    DATASETS = IMG_PATHS.keys()
    parser = ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--precision", type=int, default=16, help="Precision 16 or 32 bits")
    parser.add_argument("--copies", type=int, default=50, help="Number of image copies")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Which dataset to use", required=True)
    parser.add_argument("--method", type=str, choices=METHODS, help="Which method to use", required=True)
    parser.add_argument("--ckpt_path", type=str, help="Pre trained model path to start with", required=True)
    parser.add_argument("--debug", action='store_true', help="Init in debug mode")
    parser.add_argument("--fromcl", action='store_true', help="If the checkpoint is from contrastive learning pretraining")

    parser.print_help()    
    args = parser.parse_args()
    
    assert os.path.isfile(args.ckpt_path), "Checkpoint path needs to be a file"
        
    #false  by default
    loggers = False

    print("Hyperparameters")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("Setting Up datasets and augmentations...")
    data_transforms = misc.get_data_transforms(args.method)
    print(data_transforms)
    
    if args.dataset.startswith('atlas'):
        sep = ';'
        _format='.png'
    else:
        sep = ',' 
        _format='.jpg'
    
    #Quick fix for image file format for padufes-20. File format already in csv.
    if args.dataset in ['pad-ufes-20']:
        _format = ''
    
    ds_path = IMG_PATHS[args.dataset]
    labels_path = LABELS_PATHS[args.dataset]
    
    print(ds_path, labels_path)
    test_ds = misc.AugmentOnTest(CSVDatasetWithName(imgs_folder=ds_path, labels_csv=labels_path,
                                                    sep = sep, _format = _format,
                                                    transforms=data_transforms['test']), args.copies)

    test_loader = DataLoader(test_ds, batch_size=args.copies, shuffle=False, 
                             pin_memory=True, num_workers=args.workers)
    
    
    if not args.fromcl:
        name_exp = f'best_{args.method}_fine_tuning'
        method = args.method
    else:
        #adjust according to your ckpt_path
        name_exp = args.ckpt_path.split('/')[2]
        method = 'SimCLR' if "SimCLR" in name_exp else "SupCon"
    
    
    MODEL_CLASS = ISICFineTuner if args.fromcl else SSLModel
    
    print(f"======== Loading model from: {args.ckpt_path} -- Model Class: {MODEL_CLASS.__name__} ========")
    model = MODEL_CLASS.load_from_checkpoint(args.ckpt_path)
    
    name_exp += f"_lr_{model.params.lr}"
    
    
    print(f"======== Running with {args.dataset.upper()} Dataset ========")
    print(f"======== Experiment Name: {name_exp} \t Method: {method} ========")
    
    if not args.debug:
        print("======== Setting Up loggers and callbacks ========")

        #create the comet logger
        comet_logger = CometLogger(
            api_key=os.getenv("COMET_API_KEY"),
            project_name='top5-test',
            workspace=os.getenv("COMET_WORKSPACE"),
            experiment_name=name_exp
        )

        
        comet_logger.experiment.log_parameters(args)
        comet_logger.experiment.log_parameter('test_aug', 
                                              str({'test_aug': data_transforms['test']}))
        
        #log the code for visual inspection
        comet_logger.experiment.log_code(folder='/utils/')
        comet_logger.experiment.log_code(folder='/models/')
        comet_logger.experiment.log_code(file_name='custom_dataset.py')
        comet_logger.experiment.log_other('ds_path', ds_path)
        comet_logger.experiment.log_other('labels_path', labels_path)
        
        comet_logger.experiment.add_tag(args.dataset)
        comet_logger.experiment.add_tag(method)
        
        loggers = [comet_logger]
        
         #Log dataset sample images to Comet
        num_samples = len(test_ds)
        for _ in range(10):
            value = np.random.randint(0, num_samples)
            (img, label), name = test_ds[value]
            img = img.permute(1,2,0).numpy()
            comet_logger.experiment.log_image(img, name=f"[TEST]{name}-GT:{label}")
    
    
    
    tester = pl.Trainer(gpus=args.gpus,
                        precision=args.precision,
                        logger=loggers,
                        fast_dev_run=args.debug,
                        num_sanity_val_steps=-1
                        )
    
    print(f"======== Testing ========")
    tester.test(model, test_dataloaders=test_loader)

