# import pdb
import numpy as np
import sys
import os
import csv
import pickle
import random
import pandas as pd
from HNdataPatch_FeatureExtr import get_dataloader, get_statistics
# ----- For the 3D CNN
import torch
from torch import optim
import torch.nn as nn
from torch.optim import lr_scheduler
# from torchsummary import summary
from torchinfo import summary

from opts_pt_AE import parse_opts
from model_clipara_FeatureExtr import generate_model
from utils import WriteLogger
import pandas ,pytz

from train_3DFeatureExtr import train_epoch
from val_3DFeatureExtr import val_epoch
from test_3DFeatureExtr import test
from torch.utils.tensorboard import SummaryWriter
from losses_3DFeatureExtr import CombinedLoss

# Set seed for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def main():
    opt = parse_opts()
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    # opt.result_path = os.path.join(os.getcwd(), opt.result_path)
    losses_str = '_'.join(opt.losses)

    opt.data_path = opt.data_path_linux
    opt.result_path = opt.result_path_linux + '_test_good_normalization_model' + str(opt.model) + '_inputtype' + str(opt.input_type)+ '_inputmodality' + str(opt.input_modality)+'_fold' + str(opt.fold) \
                          + '_lr' + str(opt.learning_rate) + '_optim_' + str(opt.optimizer)+ '_bs' + str(opt.batch_size) \
                          + '_z_size' + str(opt.z_size) + '_md' + str(opt.model_depth) + '_sepconv_' + losses_str + '_' + opt.lr_scheduler
  
    ValidDataInd = pickle.load(open('/data/pg-dl_radioth/scripts/MultilabelLearning_OPC_Radiomics/OPC-Radiomics/ValidDataInd_clinical.d', 'rb'))

    opt.TBlog_dir = os.path.join(opt.result_path, opt.TBlog_dir)
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.result_path, opt.resume_path)
    opt.val_images_path = os.path.join(opt.result_path, opt.val_images_path)
    opt.test_images_path = os.path.join(opt.result_path, opt.test_images_path)

    # NEW
    # Create folder if it does not exist
    for p in [opt.result_path, opt.TBlog_dir, opt.resume_path, opt.val_images_path, opt.test_images_path]:
        if p and (not os.path.exists(p)):
            os.mkdir(p)
            
    writer = SummaryWriter(log_dir=opt.TBlog_dir)

    # ----update sample size and sample duration
    if opt.input_type ==3:
        opt.sample_size = 64
        opt.sample_duration = 32
    else:
        opt.sample_size = 144
        opt.sample_duration =72
    opt.extra_featuresize = 37
    opt.HNSCC_ptnum =606
    
    opt.ValidDataInd = ValidDataInd

    if opt.input_type == 2 or opt.input_type == 3:
        opt.input_channel = 2  
    else:
        opt.input_channel = 1
        
    if opt.input_modality == 2:
        if opt.input_type == 3:
           opt.input_channel =4
        else:       
           opt.input_channel += 1

    model, parameters = generate_model(opt)
    txt = summary(model=model, input_size=(opt.input_channel, opt.sample_duration, opt.sample_size, opt.sample_size))
	
    print(txt)    
    # pdb.set_trace()

    # save model architecture
    file = open(opt.result_path + '/model.txt', 'a+')
    file.write(str(txt))
    file.close()
    
    # save the config values 
    file = open(opt.result_path + '/parameter.txt', 'a+')
    file.write(str(opt))
    file.close()

    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    criterion = CombinedLoss(opt)
    #criterion_D = nn.BCEWithLogitsLoss()

    if not opt.no_cuda:
        criterion = criterion.cuda()

    # pdb.set_trace()
    train_logger = WriteLogger(
        os.path.join(opt.result_path, 'train.log'),
        ['epoch', 'g_loss', 'l1', 'ssim', 'ssim_piqa', 'mse', 'kld', 'adv','dice','d_loss', 'lr'])
    train_batch_logger = WriteLogger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'g_loss', 'l1', 'ssim', 'ssim_piqa', 'mse', 'kld', 'adv','dice','d_loss','lr'])
    val_logger = WriteLogger(
        os.path.join(opt.result_path, 'val.log'), ['epoch', 'g_loss', 'l1', 'ssim', 'ssim_piqa', 'mse', 'kld', 'adv','dice','d_loss'])
    test_logger = WriteLogger(
        os.path.join(opt.result_path, 'test.log'), ['epoch', 'g_loss', 'l1', 'ssim', 'ssim_piqa', 'mse', 'kld', 'adv','dice','d_loss'])

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    # opt.optimizer='adam'
    if opt.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

    elif opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            parameters,
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay)

    # a nice example on the lr_scheduler:https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/
    if opt.lr_scheduler == 'reducelr':
      scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=opt.reducelr_factor, patience=opt.lr_patience)
    elif opt.lr_scheduler == 'cycliclr':
      if opt.optimizer.lower() == 'adam':
          cycle_momentum = False
      else:
          cycle_momentum = True
      scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=20, step_size_down=1000, cycle_momentum=cycle_momentum)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
	    
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('Run the code')

    ptlist = range(opt.HNSCC_ptnum)
    
    trainval_list, test_list, all_list = Train_test_split(ptlist,ValidDataInd)
    
    train_list = random.sample(list(trainval_list),int(len(trainval_list)*0.8))
    val_list = list(set(trainval_list) - set(train_list))
    
    #train_dataloader = get_dataloader(opt, list(train_list), batch_size=opt.batch_size, shuffle=True, DataAug=True)
    train_dataloader = get_dataloader(opt, list(trainval_list), batch_size=opt.batch_size, shuffle=True, DataAug=True)
   
    # NEW
    val_dataloader = get_dataloader(opt, list(val_list), batch_size=opt.batch_size, shuffle=False, DataAug=False)
    test_loader = get_dataloader(opt, list(test_list), batch_size=opt.batch_size, shuffle=False, DataAug=False)

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            # pdb.set_trace()
            # train_loss, train_l1, train_ssim, train_ssim_piqa, train_mse, train_kld = 
            train_epoch(i, train_dataloader, model, criterion , optimizer, scheduler, opt, train_logger, train_batch_logger, writer)

        if not opt.no_val:
            # val_loss, val_l1, val_ssim, val_ssim_piqa, val_mse, val_kld = 
            scheduler = val_epoch(i, val_dataloader, model ,criterion, scheduler, opt, val_logger, writer)
    
    # just for get image, don't perform other test        
    latent_feature_test = test(1, test_loader, model,criterion, opt, test_logger, writer)            
    '''
    latent_feature_test = test(1, test_loader, model, criterion, opt, test_logger, writer)    
    latent_feature_test = np.squeeze(np.array(latent_feature_test.cpu()))
    latent_feature_test_csv = pd.DataFrame(latent_feature_test)
    latent_feature_test_csv['index'] = test_list
    latent_feature_test_csv.set_index('index')
    latent_feature_test_csv.to_csv(opt.result_path+'/latent_feature_test.csv')
     
    all_dataloader = get_dataloader(opt, list(all_list), batch_size=opt.batch_size, shuffle=False, DataAug=False)
    
    latent_feature = test(2, all_dataloader, model, criterion, opt, test_logger, writer)
    latent_feature = np.squeeze(np.array(latent_feature.cpu()))
    l_max , l_min = latent_feature.max(axis=0),latent_feature.min(axis=0)
    latent_feature = (latent_feature - l_min)/(l_max - l_min)
    latent_feature_csv = pd.DataFrame(latent_feature)
    latent_feature_csv['index'] = all_list
    latent_feature_csv.set_index('index')
    latent_feature_csv.to_csv(opt.result_path+'/latent_feature.csv')

    max_min_data = opt.result_path+'/l_max_min.d'
    pickle.dump((l_max,l_min),open(max_min_data,'wb'))
    '''
def Train_test_split(pt_list,ValidDataInd):

	ValidDataInd = np.asarray(ValidDataInd).astype(int)
	all_list = np.asarray(pt_list)[np.where(ValidDataInd>0)[0]]
	trainval_list = np.asarray(pt_list)[list(set(np.where(ValidDataInd<200)[0])- set(np.where(ValidDataInd==0)[0])) ]
	test_list = np.asarray(pt_list)[np.where(ValidDataInd>199)[0]]
	
	return trainval_list, test_list, all_list
