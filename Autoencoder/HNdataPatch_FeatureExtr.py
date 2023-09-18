"""
This script contains the data preparation for Volume Patch data that is already extracted from the original data
"""
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from torchvision import transforms
from operator import itemgetter
from scipy import ndimage
from scipy.stats import zscore
import pickle
from scipy import ndimage
from skimage.transform import resize

def GetDataDirs(data_path, data_list, valid_dataInd):
    data_dirs = []
    for i in data_list:
        data_dirs.append(data_path+'/VolPatch_' + str(int(valid_dataInd[i])) + '_data.d')

    return data_dirs

def ReadCTData(opt, data_dirs, EndPoint, EPind_list, data_list, input_transform=None, index=0):
    """
    This function is only used for Jupyter Notebook to read the CT data
    (incl. PTV masks, clinical parameters and outcome endpoints) for illustration.
    """
    label = np.zeros((len(EPind_list), len(data_list)), dtype=int)

    EPind_list = EPind_list
    for i in range(0, len(EPind_list)):
        label_singleEP = np.array(itemgetter(*data_list)(EndPoint[EPind_list[i]]))
        label[i] = label_singleEP

    VolPatch = pickle.load(open(data_dirs[index], 'rb'))
    CTPatch = VolPatch[0]
    MaskPatch = VolPatch[1]
    LabelPatch = []
    for EPind in EPind_list:
        LabelP = VolPatch[EPind+1]  # the difference in denotation of the end points in different files(EndPoints.d,VolPatch_0_data.d)
        LabelPatch.append(LabelP) 
    CliPara = GetClinicalVec(VolPatch[11:])

    if opt.input_type == 0:
        VolInput = CTPatch
        VolInput = np.expand_dims(VolInput, axis=0)
    elif opt.input_type == 1:
        VolInput = CTPatch*MaskPatch  # masking the tumor region with the delineation
        VolInput = np.expand_dims(VolInput, axis=0)
    else:
        VolInput = np.concatenate((np.expand_dims(CTPatch, axis=0), np.expand_dims(MaskPatch, axis=0)), axis=0)

    if input_transform:
        imgs = input_transform(VolInput)
    else:
        imgs = VolInput

    return imgs, label, CliPara


def GetClinicalVec(CliParaTup):
    CliParaVec = list(np.asarray(CliParaTup[0:2]))
    for i in range(2, len(CliParaTup)):
        CliParaVec.extend(CliParaTup[i])

    return np.asarray(CliParaVec)


class TargetToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __call__(self, label):
        return torch.tensor(label, dtype=torch.FloatTensor)


class RTInputNormalize(object):
    """	
    Normalize the data ([0, 1]) by the maximal intensity or the bone structure intensity
    (BoneGray=(1800-(-1024))/1, RescaleSlope=1, RescaleIntercept=-1024)
    correct it for different dataset
    """	
    def __call__(self, input):
        '''
        BoneGray = 2824.0
        # Map minimum of input to 0
        input_min = input.min()    
        print("\tinput.min() before:", input_min)
        print("\tinput.max() before:", input.max())
        input = input - input_min
        BoneGray_shifted = BoneGray - input_min
        if (input_min == -32768) or (input_min == -4275):  # note: if input_min == -32768, then somehow (input - input_min).min() = input_min.min() = -32768...
            input = input.astype(np.uint16)
            BoneGray_shifted = BoneGray_shifted.astype(np.uint16)
        print("\tinput.min() after:", input.min())
        print("\tinput.max() after (1):", input.max())
        input = input / min(input.max(), BoneGray_shifted)
        print("\tinput.max() after (2):", input.max())
        input[input > 1] = 1
        print("\tinput.max() after (3):", input.max())
        '''
        input[input>1] = 1
        input[input<0] = 0
        
        
        return input


class VolumeDataAug(object):
    """flip the volumetric data horizontally or verticallya"""
    def __init__(self, FlipDirection=None, RotateAngle=None):
        self.FlipDirection = FlipDirection
        self.RotateAngle = RotateAngle

    def __call__(self,input):
        # pdb.set_trace()
        if self.FlipDirection is not None:
            if self.FlipDirection == 0:  # 'Horizontal'
                input = np.flip(input, axis=3).copy()
            elif self.FlipDirection == 1:  # 'Vertical'
                input = np.flip(input, axis=2).copy()

        if self.RotateAngle is not None:
            input = ndimage.rotate(input, self.RotateAngle, axes=(2, 3), reshape=False).copy()

        return input


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        return torch.from_numpy(image).type(torch.FloatTensor)


class dataset_volpatch(Dataset):
    def __init__(self, opt, data_list, input_transform=None):

        self.opt = opt
        self.data_dirs = GetDataDirs(opt.data_path, data_list, opt.ValidDataInd)
        self.input_transform = input_transform

    def __getitem__(self, index):

        VolPatch = pickle.load(open(self.data_dirs[index], 'rb'))
        #CTPatch = VolPatch[0][10:82,18:162,18:162]
        CTPatch = VolPatch[0]
        #print (CTPatch.max(), CTPatch.min())
        #print (CTPatch.max(), CTPatch.min())
        
        if self.opt.HU_trans: # True for OPC-Radiomics, False for UMCG OPC
          # new
          if CTPatch.max() > 10000:
            CTPatch = CTPatch 
          else:
            CTPatch = CTPatch - 1024 # for OPC-Radiomics
            #CTPatch = CTPatch  # for UMCG OPC , external
        
        CTPatch[CTPatch > 200.] = 200.
        CTPatch[CTPatch < -200.] = -200.
        
        #print (CTPatch.max(),CTPatch.min())
        #CTPatch[CTPatch > 1024.] = 1024.
        #CTPatch[CTPatch > 200.] = 200.
        #CTPatch[CTPatch < -1024.] = -1024.
        #CTPatch[CTPatch < -200.] = -200.
        #CTPatch = zscore(np.reshape(CTPatch,[-1]))
        #CTPatch = np.reshape(CTPatch,[144,144,144])
        
        
        #CTPatch =(CTPatch - np.min(CTPatch))/(np.max(CTPatch) - np.min(CTPatch))
        CTPatch =(CTPatch - (-200))/(np.max(CTPatch) - (-200))
        
        #print (CTPatch.max(),CTPatch.min())
        
        #MaskPatch = VolPatch[1][10:82,18:162,18:162]
        MaskPatch = VolPatch[1]
        #MaskPatch = ndimage.binary_dilation(MaskPatch).astype(MaskPatch.dtype)

        
        if self.opt.input_modality == 0:
            VolInput = np.expand_dims(CTPatch, axis=0)
                     
        if self.opt.input_type == 0:
            VolInput = VolInput
        elif self.opt.input_type == 1:
            VolInput = VolInput * np.expand_dims(MaskPatch, axis=0)  # masking the tumor region with the delineation            
        elif self.opt.input_type == 2:
            VolInput = np.concatenate((VolInput, np.expand_dims(MaskPatch, axis=0)), axis=0)           
        elif self.opt.input_type == 3:   
            VolInput = VolInput * np.expand_dims(MaskPatch, axis=0)
            x_non0,y_non0,z_non0 = np.nonzero(MaskPatch)
            x_min, x_max ,y_min, y_max,z_min, z_max = np.min(x_non0),np.max(x_non0),np.min(y_non0),np.max(y_non0),np.min(z_non0),np.max(z_non0)
            #x_center , y_center, z_center  = int((x_max+x_min)/2), int((y_max+y_min)/2), int((z_max+z_min)/2)
            #print (x_min, x_max ,y_min, y_max,z_min, z_max,x_center , y_center, z_center)
            
            volinput = VolInput[0:VolInput.shape[0],x_min:x_max+1,y_min:y_max+1,z_min:z_max+1]
            volinput1 = resize(volinput,(volinput.shape[0],32,64,64))
            
            volinput2 = np.zeros_like(VolInput)
            volinput2[0:VolInput.shape[0],45-int(volinput.shape[1]/2):45-int(volinput.shape[1]/2)+volinput.shape[1],91-int(volinput.shape[2]/2):91-int(volinput.shape[2]/2)+volinput.shape[2],91-int(volinput.shape[3]/2):91-int(volinput.shape[3]/2)+volinput.shape[3]] = volinput
            volinput2 = volinput2[0:VolInput.shape[0],45-16:45+16,91-32:91+32,91-32:91+32]

            VolInput = np.concatenate((volinput1, volinput2), axis=0)

        if self.input_transform:
            imgs = self.input_transform(VolInput)
            # imgs.unsqueeze_(0)
        else:
            imgs = VolInput        

        return imgs

    def __len__(self):
        # pdb.set_trace()
        return len(self.data_dirs)


def get_dataloader(opt, data_list, batch_size=4, shuffle=False, DataAug=False):
    # pdb.set_trace()
    dicom_transform = transforms.Compose([RTInputNormalize(), ToTensor()])
    flip_transform = transforms.Compose([VolumeDataAug(FlipDirection=np.random.randint(2, size=1)),
                                         RTInputNormalize(),
                                         ToTensor()])
    rotation_transform = transforms.Compose([VolumeDataAug(RotateAngle=int(np.random.randint(-45, 45, size=1))),
                                             RTInputNormalize(),
                                             ToTensor()])  #, transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])
    dataset = dataset_volpatch(opt, data_list, input_transform=dicom_transform)

    if DataAug:
        dataset_rot = dataset_volpatch(opt, data_list, input_transform=rotation_transform)
        dataset_flip = dataset_volpatch(opt, data_list, input_transform=flip_transform)
        dataset = ConcatDataset([dataset, dataset_rot, dataset_flip])
        
    dataloader = DataLoader(dataset=dataset,
                            num_workers=0,
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader
    
        
def get_statistics(opt, csv_writer, data_list, mode, shuffle=False):
    dataset = dataset_volpatch(opt, data_list)
        
    for inputs_name in dataset.data_dirs:
        index = dataset.data_dirs.index(inputs_name)
        inputs = dataset.__getitem__(index)
        
        print("inputs_name:", inputs_name)
        print("dataset.data_dirs[index]:", dataset.data_dirs[index])
        assert inputs_name == dataset.data_dirs[index]
        
        inputs_min = inputs.min()
        inputs_max = inputs.max()
        
        csv_writer.writerow([inputs_name, mode, inputs_min, inputs_max])


