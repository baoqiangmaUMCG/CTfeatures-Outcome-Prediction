import os
import torch
import time
from torch import nn
from sklearn.metrics import mean_squared_error
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_binary_multilabel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math


def val_epoch(epoch, data_loader, model, criterion, scheduler, opt,
              epoch_logger, writer):
    print('Validate at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    g_losses = AverageMeter()
    l1 = AverageMeter()
    ssim = AverageMeter()
    ssim_piqa = AverageMeter()
    mse = AverageMeter()
    kld = AverageMeter()
    dice = AverageMeter()

    end_time = time.time()
    num_batches = len(data_loader)
    
    # fig = plt.figure(figsize = (num_batches**2, opt.batch_size**2))
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
                 recons, outputs_latent = model(inputs)
              
                 g_loss_dict = criterion(inputs=inputs, recons=recons)

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
            
            # Plot inputs vs recons
            if opt.input_type ==3:
               slice_num = 16
            else: 
               slice_num = 36
            for j in range(inputs.shape[0]):  # use inputs.shape[0] instead of opt.batch_size, because the last batch (drop_last = False) may contain less samples
                # Input
                axarr[j, 2*i] = plt.subplot(gs[j, 2*i])
                axarr[j, 2*i].imshow(inputs[j, 0, slice_num, :, :].cpu().detach(), cmap=plt.cm.gray)
                axarr[j, 2*i].set_xticklabels([])
                axarr[j, 2*i].set_yticklabels([])
                    
                # Reconstructed
                axarr[j, 2*i+1] = plt.subplot(gs[j, 2*i+1])
                axarr[j, 2*i+1].imshow(recons[j, 0, slice_num, :, :].cpu().detach(), cmap=plt.cm.gray)
                axarr[j, 2*i+1].set_xticklabels([])
                axarr[j, 2*i+1].set_yticklabels([])
            
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
    plt.savefig(os.path.join(opt.val_images_path, 'epoch_{}.png'.format(epoch)))
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
        'dice': dice.avg,

    })
    
    # For ReduceLROnPlateau
    if opt.lr_scheduler == 'reducelr':
        scheduler.step(g_losses.avg)
    
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
    return scheduler  # losses.avg, l1.avg, ssim.avg, ssim_piqa.avg, mse.avg, kld.avg
