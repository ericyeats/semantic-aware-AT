import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data
from core.data import SEMISUP_DATASETS, SCORE_DATASETS

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train
from core.utils import Trainer
from core.utils import seed

from gowal21uncovering.utils import WATrainer


# Setup

parse = parser_train()
parse.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')

# extra args for score-based constraint
parse.add_argument('--standard_pseudo', action='store_true', help='Only perform standard training on pseudo data')
parse.add_argument('--score_matching', action='store_true', help='Perform score matching with score data')
parse.add_argument('--gamma', help='Score matching regularization strength', type=float, default=0.0)
parse.add_argument('--time', help='Time in [0, 1] to which data should be diffused for average scores. default 0.1', type=float, default=0.1)
parse.add_argument('--n_mc_samples', type=int, default=20, help='Number of samples for average score calculation')
parse.add_argument('--verbose', action='store_true')
parse.add_argument('--standard', action='store_true')
parse.add_argument('--random_proj', action='store_true', help='Baseline with random score projection vector')

args = parse.parse_args()
# assert args.data in SEMISUP_DATASETS, f'Only data in {SEMISUP_DATASETS} is supported!'


DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)


info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_ADV_EPOCHS = args.num_adv_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_ADV_EPOCHS = 1

# To speed up training
torch.backends.cudnn.benchmark = True


# Load data
if args.score_matching:
    assert args.data in SCORE_DATASETS

seed(args.seed)
val = args.data in SEMISUP_DATASETS or args.data in SCORE_DATASETS
loaded_data = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, use_consistency=args.consistency, shuffle_train=True, 
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, validation=val, time=args.time, n_mc_samples=args.n_mc_samples
)
if val:
    train_dataset, test_dataset, eval_dataset, train_dataloader, test_dataloader, eval_dataloader = loaded_data
    del train_dataset, test_dataset, eval_dataset
else:
    train_dataset, test_dataset, train_dataloader, test_dataloader = loaded_data
    del train_dataset, test_dataset
    eval_dataloader = test_dataloader



# Adversarial Training

seed(args.seed)
if args.tau:
    print ('Using WA.')
    trainer = WATrainer(info, args)
else:
    trainer = Trainer(info, args)
last_lr = args.lr


if NUM_ADV_EPOCHS > 0:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)*100))
    
    old_score = [0.0, 0.0]
    logger.log('RST Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    trainer.init_optimizer(args.num_adv_epochs)
    test_adv_acc = 0.0    

if args.resume_path:
    start_epoch = trainer.load_model_resume(os.path.join(args.resume_path, 'state-last.pt')) + 1
    logger.log(f'Resuming at epoch {start_epoch}')
else:
    start_epoch = 1

for epoch in range(start_epoch, NUM_ADV_EPOCHS+1):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    
    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
    
    res = trainer.train(train_dataloader, epoch=epoch, adversarial=(not args.standard), verbose=args.verbose)
    test_acc = trainer.eval(test_dataloader)
    
    logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))
    if 'clean_acc' in res:
        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
    else:
        logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))
    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    epoch_metrics.update({'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''})
    
    if not (args.standard or args.score_matching):
        if epoch % args.adv_eval_freq == 0 or epoch == NUM_ADV_EPOCHS:        
            test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
            logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, 
                                                                                    test_adv_acc*100))
            epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
        else:
            logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc']*100))
    
    eval_adv_acc = trainer.eval(eval_dataloader, adversarial=True)
    logger.log('Adversarial Accuracy-\tEval: {:.2f}%.'.format(eval_adv_acc*100))
    epoch_metrics['eval_adversarial_acc'] = eval_adv_acc
    if eval_adv_acc >= old_score[1]:
        old_score[0], old_score[1] = test_acc, eval_adv_acc
        trainer.save_model(WEIGHTS)

    # trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))
    if epoch % 10 == 0:
        trainer.save_model_resume(os.path.join(LOG_DIR, 'state-last.pt'), epoch) 
    if epoch % 400 == 0:
        shutil.copyfile(WEIGHTS, os.path.join(LOG_DIR, f'weights-best-epoch{str(epoch)}.pt'))

    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    # metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True) ### NOTE - DEPRECATED
    metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
    metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)

    
    
# Record metrics

train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)
logger.log('\nTraining completed.')
logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
if NUM_ADV_EPOCHS > 0 and (not args.standard):
    logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tEval: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100)) 

logger.log('Script Completed.')
