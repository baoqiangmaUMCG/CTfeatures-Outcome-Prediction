import torch
from torch import nn
from ssim import SSIM3D
from piqa import SSIM
from torch.nn import functional as F

class DiceLoss(nn.Module): 
    """
    This class generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    def __init__(self, logits=True):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.logits =logits
    def forward(self,pred,target):
        # have to use contiguous since they may from a torch.view op
        pred_flat = pred.contiguous().view(-1)
        
        if self.logits:
            pred_flat = nn.Sigmoid()(pred_flat)

        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()

        A_sum = torch.sum(pred_flat * pred_flat)
        B_sum = torch.sum(target_flat * target_flat)
        loss = 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth) )
        
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, opt,lossfun1=DiceLoss()):
        super(CombinedLoss, self).__init__()
        self.opt = opt
        if opt.input_type ==3:
            self.ssim_piqa = SSIM(n_channels=32, value_range=1.).cuda()
        else:   
            self.ssim_piqa = SSIM(n_channels=72, value_range=1.).cuda()
        self.dice = lossfun1

    # NEW
    def forward(self, **kwargs):
        """
            :param inputs: batch of true 3D data
            :param recons: batch of reconstructed 3D data (from inputs data)
            :param logits: batch of reconstructed 3D data classification by discriminator (from inputs data)
        
        vae: 
            Kullback-Leibler divergence:
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
            :param mu: mean (output encoder)
            :param log_var: log variance (output encoder)
        """
        kld_weight = torch.Tensor([0])
        kld_loss = torch.Tensor([0])
        
        if (self.opt.model == 'vae') or (self.opt.model == 'resnet_vae') or (self.opt.model == 'resnet_vae_fm'):
            kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
            
            # Apply torch.sum() over all dimensions (except first dimension of batch (B)): e.g. on (C, D, H, W) if shape = (B, C, D, H, W).
            # dims = len(kwargs['mu'].shape)
            # The following approach will lead to very large KLDs of size ~2 million , which makes training impossible (Infs/NaNs)
            # kld_loss = torch.mean(-0.5 * torch.sum(1 + kwargs['log_var'] - kwargs['mu'] ** 2 - kwargs['log_var'].exp(), dim=tuple([x for x in range(1, dims)])), dim=0)
            # else:
            # kld_loss = torch.mean(-0.5 * torch.sum(1 + kwargs['log_var'] - kwargs['mu'] ** 2 - kwargs['log_var'].exp(), dim=1), dim=0)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + kwargs['log_var'] - kwargs['mu'] ** 2 - kwargs['log_var'].exp(), dim=1))
                
        l1_loss = nn.L1Loss()(kwargs['inputs'], kwargs['recons'])
        ssim_loss = SSIM3D()(kwargs['inputs'], kwargs['recons'])
        mse_loss = nn.MSELoss()(kwargs['inputs'], kwargs['recons'])
 
        dice_loss = self.dice(kwargs['recons'], kwargs['inputs'])

        ssim_piqa_loss = 0
        print("kwargs['inputs'].min():", kwargs['inputs'].min())
        print("kwargs['inputs'].max():", kwargs['inputs'].max())
        print("kwargs['recons'].min():", kwargs['recons'].min())
        print("kwargs['recons'].max():", kwargs['recons'].max())
        for i, (x, y) in enumerate(zip(kwargs['inputs'], kwargs['recons'])):
            ssim_piqa_loss += self.ssim_piqa(x, y)  
        ssim_piqa_loss = ssim_piqa_loss / (i + 1)  # Average over batch
              
        loss = 0
        for l in self.opt.losses:
            if l == 'l1':
                loss += l1_loss
            elif l == 'ssim':
                loss += 0.5* (1 - ssim_loss)
            elif l == 'ssim_piqa':
                loss += (1 - ssim_piqa_loss)
            elif l == 'mse':
                loss += mse_loss
            elif l == 'kld':
                loss += kld_weight * kld_loss
            elif l == 'dice':
                loss += dice_loss
            else:
                raise Exception('{} is an invalid loss function!'.format(l))
                    
        return {'loss': loss, 'L1': l1_loss, 'SSIM': ssim_loss, 'SSIM_piqa': ssim_piqa_loss, 'MSE': mse_loss, 'KLD': kld_weight * kld_loss,  'DICE':dice_loss}  # -kld_loss}
