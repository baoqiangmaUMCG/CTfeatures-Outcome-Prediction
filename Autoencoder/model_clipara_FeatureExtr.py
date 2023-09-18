import torch
from torch import nn

from models_clipara import Autoencoder_resnet_3D, vanilla_vae_3D, resnet_vae_3D, Autoencoder_resnet_featuremap_3D, resnet_vae_featuremap_3D, Resnet_featuremap_pyramid_sum_Jiapan_3D, Resnet_featuremap_pyramid_sum_CIT_3D, Resnet_featuremap_pyramid_concat_Jiapan_3D, Resnet_featuremap_pyramid_concat_CIT_3D,Resnet_featuremap_pyramid_sum_CIT_3D_64
from models_clipara.Autoencoder_resnet_3D import get_fine_tuning_parameters


def generate_model(opt):
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'vae', 'resnet_vae', 'resnet_fm', 'resnet_vae_fm',
        'resnet_fm_pyramid_sum_Jiapan', 'resnet_fm_pyramid_sum_CIT', 'resnet_fm_pyramid_concat_Jiapan', 'resnet_fm_pyramid_concat_CIT'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = Autoencoder_resnet_3D.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                z_size=opt.z_size)

        elif opt.model_depth == 18:
            model = Autoencoder_resnet_3D.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                z_size=opt.z_size)

        elif opt.model_depth == 34:
            model = Autoencoder_resnet_3D.resnet34(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn, input_channel=opt.input_channel,
                z_size=opt.z_size)

        elif opt.model_depth == 50:
            model = Autoencoder_resnet_3D.resnet50(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                z_size=opt.z_size)

        elif opt.model_depth == 101:
            model = Autoencoder_resnet_3D.resnet101(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                z_size=opt.z_size)

        elif opt.model_depth == 152:
            model = Autoencoder_resnet_3D.resnet152(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                z_size=opt.z_size)

        elif opt.model_depth == 200:
            model = Autoencoder_resnet_3D.resnet200(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                z_size=opt.z_size)

    elif opt.model == 'vae':
        model = vanilla_vae_3D.VanillaVAE(input_channel=opt.input_channel, z_size=opt.z_size)
    
    elif opt.model == 'resnet_vae':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet_vae_3D.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                z_size=opt.z_size)

        elif opt.model_depth == 18:
            model = resnet_vae_3D.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                z_size=opt.z_size)

    elif opt.model == 'resnet_fm':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = Autoencoder_resnet_featuremap_3D.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)

        elif opt.model_depth == 18:
            model = Autoencoder_resnet_featuremap_3D.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)

    elif opt.model == 'resnet_vae_fm':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet_vae_featuremap_3D.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)

        elif opt.model_depth == 18:
            model = resnet_vae_featuremap_3D.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)
                
    elif opt.model == 'resnet_fm_pyramid_sum_Jiapan':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = Resnet_featuremap_pyramid_sum_Jiapan_3D.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)

        elif opt.model_depth == 18:
            model = Resnet_featuremap_pyramid_sum_Jiapan_3D.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)

    elif opt.model == 'resnet_fm_pyramid_sum_CIT':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
          if opt.input_type == 3:
            model = Resnet_featuremap_pyramid_sum_CIT_3D_64.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                output_channel=opt.input_channel)
              
          else:
            model = Resnet_featuremap_pyramid_sum_CIT_3D.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                output_channel=opt.input_channel)
              
        elif opt.model_depth == 18:
          if opt.input_type == 3:
            model = Resnet_featuremap_pyramid_sum_CIT_3D_64.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                output_channel=opt.input_channel)      
          else:
            model = Resnet_featuremap_pyramid_sum_CIT_3D.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel,
                output_channel=opt.input_channel)
                
    elif opt.model == 'resnet_fm_pyramid_concat_Jiapan':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = Resnet_featuremap_pyramid_concat_Jiapan_3D.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)

        elif opt.model_depth == 18:
            model = Resnet_featuremap_pyramid_concat_Jiapan_3D.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)
                
    elif opt.model == 'resnet_fm_pyramid_concat_CIT':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = Resnet_featuremap_pyramid_concat_CIT_3D.resnet10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)

        elif opt.model_depth == 18:
            model = Resnet_featuremap_pyramid_concat_CIT_3D.resnet18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                extra_featuresize=opt.extra_featuresize,
                actfn=opt.model_actfn,
                input_channel=opt.input_channel)
    
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        
    return model, model.parameters()
