import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='HNSCC',
        type=str,
        help='Dataset')
    parser.add_argument(
        '--result_path_win',
        default='Z:\\Jiapan\\Radiotherapy\\HNSCC\\Results\\results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--result_path_linux',
        default='/data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--data_path_win',
        default='Y:\\Jiapan\\PublicDatasets\\HNSCCdataset\\VolPatch_clinical',
        type=str,
        help='Data directory path')
    parser.add_argument(
        '--data_path_linux',
        default='/data/pg-dl_radioth/scripts/MultilabelLearning_OPC_Radiomics/OPC-Radiomics/VolPatch_clinical',
        type=str,
        help='Data directory path')
        
    parser.add_argument(
        '--data_test_path_linux',
        default='/data/pg-umcg_mii/pfs_prediction/data_test/VolPatch_clinical',
        type=str,
        help='Test Data directory path')
    parser.add_argument(
        '--save_stats', 
        action='store_false', 
        help='extract statistics (min, max) of each input file, and save them to csv')  # store_false = True
    parser.add_argument(
        '--sample_size',
        default=142,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=78,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--losses',
        default=['l1','mse','ssim'],
        type=list,
        help=
        'Loss functions for training: (l1 | ssim | ssim_piqa | mse | kld (only VAE, ResNetVAE and ResNetVAE_FM!))')
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str,
        help='Currently only support sgd and adam')
    parser.add_argument(
        '--lr_scheduler',
        default='reducelr',
        type=str,
        help='LR scheduler: (reducelr | cycliclr)')
    parser.add_argument(
        '--reducelr_factor',
        default=0.5,
        type=float,
        help='Scaling factor of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--lr_patience',
        default=5,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--batch_size', 
        default=6, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=80,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument(
        '--z_size',
        default=4096,
        type=int,
        help='Size of the latent space vector z.')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training, for example: save_100.pth')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    # NEW
    parser.add_argument(
        '--val_images_path',
        default='val_images',
        type=str,
        help='Save images (.png) of validation step')
    parser.add_argument(
        '--test_images_path',
        default='test_images',
        type=str,
        help='Save images (.png) of testing step')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--scale_in_test',
        default=1.0,
        type=float,
        help='Spatial scale in test')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--model',
        default='resnet_fm_pyramid_sum_CIT',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | vae | resnet_vae | resnet_fm | resnet_vae_fm | resnet_fm_pyramid_sum_Jiapan | resnet_fm_pyramid_sum_CIT | resnet_fm_pyramid_concat_Jiapan | resnet_fm_pyramid_concat_CIT)')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet(_vae) (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--model_actfn',
        default='relu',
        type=str,
        help='activation function')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')  # not sure about this argument
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--TBlog_dir',
        default='TBlog',
        type=str,
        help='save Tensor Board log')
    parser.add_argument(
        '--input_type',
        default=0,
        type=int,
        help='Different types of inputs (0 for only CT, 1 for masking CT ,2 for CT and mask are in two channels,32 for 2 64x64x64 CT patches by GTV)')  #
    parser.add_argument(
        '--input_modality',
        default=0,
        type=int,
        help='Different types of input modality (0 for only CT, 1 for only PET, 2 for CT and PET in two channels,3 for CT and PET combined in 50% pixels)')  #
    parser.add_argument(
        '--fold',
        default=1,
        type=int,
        help='fold number')  #
    parser.add_argument(
        '--sub',
        default=1,
        type=int,
        help='sub number 50 steps of 1000 steps of boostraping')  

    args = parser.parse_args()

    return args
