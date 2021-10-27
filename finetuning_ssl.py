#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import torchvision
from torchvision import transforms

import numpy as np

from sklearn import metrics

from argparse import ArgumentParser

import os

from utils import misc

from custom_dataset import Mode, ISIC2019, DatasetWithName


class SSLModel(pl.LightningModule):
    def __init__(self, params):
        super(SSLModel, self).__init__()
        
        #save the parameters as arguments
        self.params = params
        
        #save parameters for late testing
        self.save_hyperparameters()
        
        #the encoder network
        self.encoder = misc.get_model(params.method, params.ft_featmap)
        
        #TODO: parametrize this. RESNET -> 2 classes
        self.classifier = nn.Linear(2048, self.params.classes)
    
    def forward(self, x):
        #standard forwass pass. Get the latent representation for each input
        #(B, 2048) for resnet 50
        representations = self.encoder(x)
        
        return representations

    def training_step(self, train_batch, batch_idx):
        """
        Function to be executed when a batch is sampled from the dataloader in training step
        
        Args:
        train_batch: the batch itself according to the dataloader
        batch_idx: index for each sample in the current batch 
        
        """
        #get the imagens and labels
        imgs, labels = train_batch
        
        #get batch latent representation
        representations = self.forward(imgs)
        
        #calculate the logits
        logits = self.classifier(representations)
        
        #calculate the cross-entropy
        loss = F.cross_entropy(logits, labels)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        
        #return the loss for optimzation. Labels and cofidences are used in training_epoch_end
        return {'loss': loss,
                'labels': labels,
                'confidences': F.softmax(logits.detach(), dim=1)[:, 1]}
    
    def training_epoch_end(self, outputs):
        """
        Function to be executed after passing through all batches in training step
        
        Args:
        outputs: all outputs from the training_step (list by default)
        
        """
        confidence = torch.cat([x['confidences'] for x in outputs], dim=0)
        confidence = confidence.cpu().numpy()
        
        #get the cofidence of all predictions
        labels = torch.cat([x['labels'] for x in outputs], dim=0)
        labels = labels.cpu().numpy()
        
        #true vs pred
        auc = metrics.roc_auc_score(labels, confidence)
        
        self.log(f'train_AUC_epoch', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        """
        Function to be executed when a batch is sampled from the dataloader in validation step
        
        Args:
        train_batch: the batch itself according to the dataloader
        batch_idx: index for each sample in the current batch 
        
        """
        #get the images and labels
        imgs, labels = batch
        
        #get the representations and logits
        representations = self.encoder(imgs)
        logits = self.classifier(representations)
        
        #calculate the loss
        loss = F.cross_entropy(logits, labels)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        #return the labels and the confidences for calculating the auc score
        return {'labels': labels,
               'confidences': F.softmax(logits.detach(), dim=1)[:, 1]}
    
    def validation_epoch_end(self, outputs):
        """
        Executes after passing through all batches in validation step
        
        Args:
        outputs: all outputs from the validation_step (list by default)
        
        """
        #get all confidences calculated in validation_step for malign class
        confidence = torch.cat([x['confidences'] for x in outputs], dim=0)
        confidence = confidence.cpu().numpy()
        
        #store the labels for all sample in a numpy array
        labels = torch.cat([x['labels'] for x in outputs], dim=0)
        labels = labels.cpu().numpy()
            
        #calculate the auc score given the labels and confidences about the positive class
        auc = metrics.roc_auc_score(labels, confidence)
        
        self.log(f'val_AUC_epoch', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
        
    def test_step(self, batch, batch_idx):
        """
        Executes when a batch is sampled from the dataloader in test step
        
        Args:
        train_batch: the batch itself according to the dataloader
        batch_idx: index for each sample in the current batch 
        
        """
        #get tha images, labels and name of the imagefile
        (imgs, labels), names = batch
        
        representations = self.encoder(imgs)
        logits = self.classifier(representations)
        
        loss = F.cross_entropy(logits, labels)
        
        #calculate the confidence for each prediction
        scores = F.softmax(logits, dim=1)[:, 1].cpu().data.numpy()
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'labels': labels.detach().data[0],
               'name': names[0],
               'scores': scores.mean()}
        
    
    def test_epoch_end(self, outputs):
        """
        Function to be executed after passing through all batches in test step
        
        Args:
        outputs: all outputs from the test_step (list by default)
        
        """
        #variable to store all confidences
        all_scores = []
        #variable to store all labels
        all_labels = []
        #dict to store the confidence for malignant class for each sample
        preds_dict = {}
        
        all_names = [x['name'] for x in outputs]
        all_scores = [x['scores'] for x in outputs]
        model_predictions = [int(score > 0.5) for score in all_scores]
        
        for name, score in zip(all_names, all_scores):
            preds_dict[name] = score
        
        for k, v in preds_dict.items():
            print("{},{}".format(k, v))
        
        #get the labels for all preidctions
        all_labels = [x['labels'].item() for x in outputs]
        
        #true vs pred
        if np.unique(all_labels).shape[0] > 1:
            auc = metrics.roc_auc_score(all_labels, all_scores)
        else:
            auc = 0.0
        balanced_acc = metrics.balanced_accuracy_score(all_labels, model_predictions)
        
        #log the auc and balanced acc for test set
        self.log(f'test_AUC_epoch', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_balanced_ACC_epoch', balanced_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.params.lr, momentum=0.9, weight_decay=0.001)
        if self.params.opt == 'plateau':
            lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, 
                                                                                   min_lr=1e-5, patience=10),
                            'monitor': 'val_loss'}
        elif self.params.opt == 'cosine':
            lr_scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.epochs,
                                                                                    eta_min=1e-5)}
        return [optimizer], [lr_scheduler]




if __name__ == "__main__":
    AVAILABLE_METHODS = ['simclr', 'swav', 'byol', 'baseline', 'infomin', 'moco']
    opts = ['plateau', 'cosine'] 

    parser = ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer Learning Rate.")
    parser.add_argument("--precision", type=int, default=16, help="Precision 16 or 32 bits")
    parser.add_argument("--method", type=str, help=f"Method to fine-tune. Available: {AVAILABLE_METHODS}", required=True)
    parser.add_argument("--epochs", type=int, default=100, help="Max number of epochs the model should run")
    parser.add_argument("--opt", type=str, choices=opts, default='plateau', help="optimizer")
    parser.add_argument("--patience", type=int, default=22, help="Patience param for early stopping")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes to classify")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validate")
    parser.add_argument("--runs", type=int, default=1, help="Run the experiment for  times")
    parser.add_argument("--copies", type=int, default=50, help="Number of image copies in test stage")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--splits_folder", type=str, default='/splits/', help="Base folder storing train, val, and test splits in csv files")
    parser.add_argument("--ft_featmap", action='store_true', help="Fine-tuning features map. Projections is the default behaviour")
    parser.add_argument("--balanced", action='store_true', help="use balanced batches")
    parser.add_argument("--debug", action='store_true', help="Init in debug mode")

    parser.print_help()
    
    args = parser.parse_args()
    
    assert args.method in AVAILABLE_METHODS, f"Select a proper method. Options available: {AVAILABLE_METHODS}"
    
    #these are false and none by default
    loggers = False
    callbacks = None

    print("Hyperparameters")
    for k, v in vars(args).items():
        print(f"{k}: {v}")


    print("Setting Up datasets and augmentations...")

    data_transforms = misc.get_data_transforms(args.method)
    print(data_transforms)
    
    #isic 2019 datafolder
    ds_path = "/ISIC_2019_Training_Input/"
    train_ds = ISIC2019(ds_path, Mode.TRAIN, args.splits_folder, data_transforms['train'])
    val_ds = ISIC2019(ds_path, Mode.VAL, args.splits_folder, data_transforms['val'])
    test_ds = misc.AugmentOnTest(DatasetWithName(ds_path, 
                                                 Mode.TEST, 
                                                 args.splits_folder,
                                                 data_transforms['test']), args.copies)

    train_sampler = None
    if args.balanced: 
        train_sampler = sampler.WeightedRandomSampler(train_ds.sampler_weights, 
                                                      len(train_ds))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None,
                              pin_memory=True, num_workers=args.workers, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.copies, shuffle=False, 
                             pin_memory=True, num_workers=args.workers)
    
    
    for run in range(args.__dict__.get('runs', 1)):
        if not args.debug:
            split_number = 1
            print("Setting Up loggers and callbacks...")
            name_exp = f'{args.method}_split_{split_number}'
            
            if args.balanced:
                name_exp = "balanced_" + name_exp

            #path to save checkpoints and csv logging
            model_path = './'

            csv = CSVLogger(model_path, 
                            name=f'{args.method}/split_{split_number}/')


            # Learning Rate Logger
            lr_logger = LearningRateMonitor()
            # Set Early Stopping
            early_stopping = EarlyStopping('val_loss', mode='min', patience=args.patience)

            # saves checkpoints to 'dirpath' whenever 'val_loss' has a new min
            checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                                  mode='min', 
                                                  save_top_k=1,
                                                  dirpath=f'{model_path}{args.method}/split_{split_number}/version_{csv._get_next_version()}/checkpoints/',
                                                  filename='{epoch:03d}-{val_loss:.3f}-{val_AUC_epoch:.3f}')

            callbacks = [lr_logger, early_stopping, checkpoint_callback]
            loggers = [csv]


        print("Setting Up Model and PL-Trainer...")
        
        #TRAINER SETTINGS
        model = SSLModel(args)

        trainer = pl.Trainer(max_epochs=args.epochs,
                             gpus=args.gpus,
                             precision=args.precision,
                             logger=loggers,
                             callbacks=callbacks,
                             fast_dev_run=args.debug,
                             num_sanity_val_steps=-1
                            )


        trainer.fit(model, train_loader, val_loader)


        if not args.debug:
            base_path = csv._save_dir + csv._name + 'version_' + str(csv._version) 
            metrics_path = base_path + "/metrics.csv"
            param_path = base_path + "/hparams.yaml"

            print("Logging...")
            print(metrics_path, param_path, base_path)

            best_model_path = checkpoint_callback.best_model_path
            print(f"BEST MODEL FOUND AT PATH: {best_model_path}")


        #if it is in debug mode, test with the weights from the last epoch
        if args.debug:
            trainer.test(model, test_dataloaders=test_loader)
        else:
            #otherwise use the best model obtained according to the callback function
            print(f"Testing with model from path: {checkpoint_callback.best_model_path}")
            trainer.test(test_dataloaders=test_loader,
                         ckpt_path=checkpoint_callback.best_model_path)
    


