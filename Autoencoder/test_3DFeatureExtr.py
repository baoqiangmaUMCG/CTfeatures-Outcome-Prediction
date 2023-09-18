import torch
import torchvision
from torch.autograd import Variable
import time
import os
import sys
from torch import nn
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from utils import AverageMeter, calculate_accuracy,calculate_accuracy_binary_multilabel
import pdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import SimpleITK as sitk

def test(epoch, data_loader, model, criterion, opt,
                epoch_logger,writer):
    print('Test at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    g_losses = AverageMeter()
    l1 = AverageMeter()
    ssim = AverageMeter()
    ssim_piqa = AverageMeter()
    mse = AverageMeter()
    kld = AverageMeter()
    adv = AverageMeter()
    dice = AverageMeter()

    end_time = time.time()
    num_batches = len(data_loader)

    # NEW: REMOVE???
    Latent_feature_cat = torch.Tensor().cuda()
    
    # fig = plt.figure(figsize = (opt.batch_size * num_batches, opt.batch_size * num_batches))
    fig, axarr = plt.subplots(nrows = opt.batch_size, ncols = 2 * num_batches, figsize = (num_batches * 4, opt.batch_size * 2), gridspec_kw = {'wspace': 0, 'hspace': 0})
    gs = fig.add_gridspec(nrows = opt.batch_size, ncols = 2 * num_batches, hspace = 0)

    with torch.no_grad():
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            # pdb.set_trace()

            if not opt.no_cuda:
                inputs = inputs.cuda(non_blocking=True)   
            
            # NEW
            if (opt.model == 'resnet') or (opt.model == 'resnet_fm') or (opt.model == 'resnet_fm_pyramid_sum_Jiapan') or (opt.model == 'resnet_fm_pyramid_sum_CIT') or (opt.model == 'resnet_fm_pyramid_concat_Jiapan') or (opt.model == 'resnet_fm_pyramid_concat_CIT'):
                 recons, outputs_latent  = model(inputs)
                 if opt.input_type == 3:
                    outputs_latent = outputs_latent.view(outputs_latent.size(0),-1)
                    #outputs_latent = nn.AvgPool3d((int(math.ceil(opt.sample_size / 16)), int(math.ceil(opt.sample_duration / 16)), int(math.ceil(opt.sample_duration / 16))), stride=1)(outputs_latent)
                 else:
                    outputs_latent = nn.AvgPool3d((int(math.ceil(opt.sample_size / 32)), int(math.ceil(opt.sample_duration / 32)), int(math.ceil(opt.sample_duration / 32))), stride=1)(outputs_latent)
                 # for pca, get all features
                 #outputs_latent = outputs_latent.view(outputs_latent.size(0),-1)
                 print (outputs_latent.size)
                 g_loss_dict = criterion(inputs=inputs, recons=recons, )
            

            # NEW
            g_loss = g_loss_dict['loss']
            g_losses.update(g_loss.item(), inputs.size(0))
            l1.update(g_loss_dict['L1'].item(), inputs.size(0))
            ssim.update(g_loss_dict['SSIM'].item(), inputs.size(0))
            ssim_piqa.update(g_loss_dict['SSIM_piqa'].item(), inputs.size(0))
            mse.update(g_loss_dict['MSE'].item(), inputs.size(0))
            kld.update(g_loss_dict['KLD'].item(), inputs.size(0))
            dice.update(g_loss_dict['DICE'].item(), inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # NEW: REMOVE???
            Latent_feature_cat = torch.cat((Latent_feature_cat,outputs_latent), 0)
            
            
            #plt.figure()
            #plt.subplot(2,2,1)
            #plt.imshow(inputs[0,0,46,:,:].cpu().detach(),cmap=plt.cm.gray)
            #plt.subplot(2,2,2)
            #plt.imshow(recons[0,0,46,:,:].cpu().detach(),cmap=plt.cm.gray)
            #plt.subplot(2,2,3)
            #plt.imshow(inputs[0,0,46,:,:].cpu().detach(),cmap=plt.cm.gray)
            #plt.subplot(2,2,4)
            #plt.imshow(recons[0,0,46,:,:].cpu().detach(),cmap=plt.cm.gray)
            #plt.title('(left-inputs, right-reconstructed)')
            #plt.show()
            
            # Plot inputs vs recons
            if opt.input_type ==3:
               slice_num = 16
            else: 
               slice_num = 36
            for j in range(inputs.shape[0]):  # use inputs.shape[0] instead of opt.batch_size, because the last batch (drop_last = False) may contain less samples
                # Input
                axarr[j, 2*i] = plt.subplot(gs[j, 2*i])
                axarr[j, 2*i].imshow(inputs[j, 0, slice_num, :, :].cpu().detach(), cmap=plt.cm.gray,  vmin = 0, vmax  = 1)
                axarr[j, 2*i].set_xticklabels([])
                axarr[j, 2*i].set_yticklabels([])
                    
                
                # Reconstructed
                axarr[j, 2*i+1] = plt.subplot(gs[j, 2*i+1])
                axarr[j, 2*i+1].imshow(recons[j, 1, slice_num, :, :].cpu().detach(), cmap=plt.cm.gray,   vmin = 0, vmax  = 1)
                axarr[j, 2*i+1].set_xticklabels([])
                axarr[j, 2*i+1].set_yticklabels([])
                
                '''
                # feature map
                axarr[j, 2*i+1] = plt.subplot(gs[j, 2*i+1])
                axarr[j, 2*i+1].imshow(feature_maps[j, 0, 4, :, :].cpu().detach())
                axarr[j, 2*i+1].set_xticklabels([])
                axarr[j, 2*i+1].set_yticklabels([])
                '''
                
                # save input and reconstruct to nifiti
            
                inputs_channel0 = sitk.GetImageFromArray(inputs[j, 0].cpu().numpy())
                inputs_channel1 = sitk.GetImageFromArray(inputs[j, 1].cpu().numpy())
                recons_channel0 = sitk.GetImageFromArray(recons[j, 0].cpu().numpy())
                recons_channel1 = sitk.GetImageFromArray(recons[j, 1].cpu().numpy())
                
                sitk.WriteImage(inputs_channel0, os.path.join(opt.test_images_path, str(i*opt.batch_size + j) + '_inputc0.nii.gz'))
                sitk.WriteImage(inputs_channel1, os.path.join(opt.test_images_path, str(i*opt.batch_size + j) + '_inputc1.nii.gz'))
                sitk.WriteImage(recons_channel0, os.path.join(opt.test_images_path, str(i*opt.batch_size + j) + '_reconsc0.nii.gz'))
                sitk.WriteImage(recons_channel1, os.path.join(opt.test_images_path, str(i*opt.batch_size + j) + '_reconsc1.nii.gz'))
                
                
                
            # Add title
            axarr[0, 2*i].set_title("Input")
            axarr[0, 2*i+1].set_title("Reconstructed")

            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'mse {mse.val:.3f} ({mse.avg:.3f})'.format(
            #           epoch,
            #           i + 1,
            #           len(data_loader),
            #           batch_time=batch_time,
            #           data_time=data_time,
            #           loss=losses,
            #           mse=mse))

            # NEW
            n_iter = (epoch - 1) * len(data_loader) + (i + 1)
            writer.add_scalar('G_Loss/val', g_loss.item(), n_iter)
            writer.add_scalar('L1/val', g_loss_dict['L1'].item(), n_iter)
            writer.add_scalar('SSIM/val', g_loss_dict['SSIM'].item(), n_iter)
            writer.add_scalar('SSIM_piqa/val', g_loss_dict['SSIM_piqa'].item(), n_iter)
            writer.add_scalar('MSE/val', g_loss_dict['MSE'].item(), n_iter)
            writer.add_scalar('KLD/val', g_loss_dict['KLD'].item(), n_iter)
            writer.add_scalar('DICE/val', g_loss_dict['DICE'].item(), n_iter)
         
    # NEW
    plt.tight_layout()
    plt.savefig(os.path.join(opt.test_images_path, 'epoch_{}_featuremap.png'.format(epoch)))
    plt.close()
    # plt.cla()  # Clear axes
    # plt.clf()  # Clear entire current figure

    # NEW
    epoch_logger.log({
        'epoch': epoch,
        'g_loss': g_losses.avg,
        'l1': l1.avg,
        'ssim': ssim.avg,
        'ssim_piqa': ssim_piqa.avg,
        'mse': mse.avg,
        'kld': kld.avg,
        'adv': adv.avg,
        'dice': dice.avg
    })
    
     # NEW
    batch_time.reset()
    data_time.reset()
    g_losses.reset()
    l1.reset()
    ssim.reset()
    ssim_piqa.reset()
    mse.reset()
    kld.reset()
    dice.reset()
    
    # NEW
    # return losses.avg, l1.avg, ssim.avg, ssim_piqa.avg, mse.avg, kld.avg
    return Latent_feature_cat


 



