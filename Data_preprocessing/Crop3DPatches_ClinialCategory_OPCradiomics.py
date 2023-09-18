__project__   = "DAME"
__author__    = "ZKH\guoj"
__function__  = "This function is used to crop the 3D patches as well as the clinical parameters in the OPCradiomics dataset for CT-based prognostic outcome prediction" 
__version__   = "0.0.1"


import os
import pydicom as dicom
import numpy as np
import pdb
import glob
from PIL import Image,ImageDraw
import skimage
from skimage import measure 
import matplotlib.pyplot as plt
import matplotlib.image as pli
import pickle
import csv
import random
import pandas as pd
import xlsxwriter 
from scipy import ndimage


def main():

    # all pts in one variables
    # our interested variable
    #/
    # organization of the clinical parameters and endpoints
    ValidDataInd = np.zeros(606)

    OPCradiomicsdata = pd.read_csv('./opcradiomics_digits_202103.csv',header=0)

    # grouping different levels of the categorical variables
    OPCradiomicsdata['TSTAD_DEF_codes_T123VS4'] = np.where(OPCradiomicsdata['TSTAD_DEF_codes']==3,1,0)

    OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3'] = OPCradiomicsdata['NSTAD_DEF_codes']
    OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3'].loc[np.where(OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3']==1)] = 0
    OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3'].loc[np.where(OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3']==2)] = 1
    OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3'].loc[np.where(OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3']==3)] = 1
    OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3'].loc[np.where(OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3']==4)] = 1
    OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3'].loc[np.where(OPCradiomicsdata['NSTAD_DEF_codes_N01VSN2VSN3']==5)] = 2

    OPCradiomicsdata['WHO_SCORE_codes_0VS123'] = np.where(OPCradiomicsdata['WHO_SCORE_codes']==0,0,1)

    clinical_para_strat = [ 'WHO_SCORE_codes_0VS123','AGE','GESLACHT_codes','Smoking_codes','TSTAD_DEF_codes_T123VS4', 'NSTAD_DEF_codes_N01VSN2VSN3', 'P16_codes']

    event_columns_code = ['OS_code','TumorSpecificSurvival_code','MET_code','LR_code','RR_code','LRR_code','DFS_code'] # DFS here is not correct
    event_2year_columns = ['OS_2year','TumorSpecificSurvival_2year','MET_code_2year','LR_code_2year','RR_code_2year',
                            'LRR_code_2year','DFS_code_2year','ULCER_code_2year']
    uncensoring_for2year_columns = ['OS_2year_uncensoring','TumorSpecificSurvival_2year_uncensoring','MET_code_2year_uncensoring',
                                    'LR_code_2year_uncensoring','RR_code_2year_uncensoring','LRR_code_2year_uncensoring',
                                    'DFS_code_2year_uncensoring','ULCER_code_2year_uncensoring'] # 1--uncensored, 0 --censored
    survival_columns = ['TIME_OS','TIME_TumorSpecificSurvival','TIME_MET','TIME_LR','TIME_RR','TIME_LRR','TIME_DFS','TIME_ULCER']

    # MET, TIMEMET, LocationSite, LRR, TIMELRR, RRR, TIMERRR, IndexTumorDEATH, TIMEITDEATH, DFS, TIMEDFS, OverallSurvival, TIMEOS,
    # GenderVec, AgeFloat, ModalityCatVec, WHO_ScoreVec, TstageVec, NstageVec, ClinicalStageVec, TS_HPVposVec)

    OS  = OPCradiomicsdata[event_columns_code[0]]
    TSS = OPCradiomicsdata[event_columns_code[1]]
    MET = OPCradiomicsdata[event_columns_code[2]]
    LR  = OPCradiomicsdata[event_columns_code[3]]
    RR  = OPCradiomicsdata[event_columns_code[4]]
    LRR = OPCradiomicsdata[event_columns_code[5]]
    DFS = OPCradiomicsdata[event_columns_code[6]]

    TIME_OS  = OPCradiomicsdata[survival_columns[0]]
    TIME_TSS = OPCradiomicsdata[survival_columns[1]]
    TIME_MET = OPCradiomicsdata[survival_columns[2]]
    TIME_LR  = OPCradiomicsdata[survival_columns[3]]
    TIME_RR  = OPCradiomicsdata[survival_columns[4]]
    TIME_LRR = OPCradiomicsdata[survival_columns[5]]
    TIME_DFS = OPCradiomicsdata[survival_columns[6]]


    OS_2year  = OPCradiomicsdata[event_2year_columns[0]]
    TSS_2year = OPCradiomicsdata[event_2year_columns[1]]
    MET_2year = OPCradiomicsdata[event_2year_columns[2]]
    LR_2year  = OPCradiomicsdata[event_2year_columns[3]]
    RR_2year  = OPCradiomicsdata[event_2year_columns[4]]
    LRR_2year = OPCradiomicsdata[event_2year_columns[5]]
    DFS_2year = OPCradiomicsdata[event_2year_columns[6]]

    OS_2year_uncen  = OPCradiomicsdata[uncensoring_for2year_columns[0]]
    TSS_2year_uncen = OPCradiomicsdata[uncensoring_for2year_columns[1]]
    MET_2year_uncen = OPCradiomicsdata[uncensoring_for2year_columns[2]]
    LR_2year_uncen  = OPCradiomicsdata[uncensoring_for2year_columns[3]]
    RR_2year_uncen  = OPCradiomicsdata[uncensoring_for2year_columns[4]]
    LRR_2year_uncen = OPCradiomicsdata[uncensoring_for2year_columns[5]]
    DFS_2year_uncen = OPCradiomicsdata[uncensoring_for2year_columns[6]]


    WHO_SCORE_codes_0VS123      = OPCradiomicsdata[clinical_para_strat[0]]
    Age                         = OPCradiomicsdata[clinical_para_strat[1]]
    Gender                      = OPCradiomicsdata[clinical_para_strat[2]]
    Smoking                     = OPCradiomicsdata[clinical_para_strat[3]]
    TSTAD_DEF_codes_T123VS4     = OPCradiomicsdata[clinical_para_strat[4]]
    NSTAD_DEF_codes_N01VSN2VSN3 = OPCradiomicsdata[clinical_para_strat[5]]
    P16_codes                   = OPCradiomicsdata[clinical_para_strat[6]]

    AllPaRTFile = pickle.load(open('./AllScanROIFile_OPCRadiomics.d','rb'))

    # the following parameters are determined by the umcg dataset
    crop_sz = 90
    crop_slice = 46
    obj2dis = 'GTV'
    resized_spacing = [2,1,1] # the same as the UMCG dataset, [slicethickness,pixelspacing,pixelspacing]

    ptnum = 0
    # Excluded pts No.
    ExcPT = [0,3] 
    EndPoints = (OS,TIME_OS,TSS,TIME_TSS,MET,TIME_MET,LR,TIME_LR,RR,TIME_RR,LRR,TIME_LRR,DFS,TIME_DFS,
                    OS_2year,OS_2year_uncen,TSS_2year,TSS_2year_uncen,MET_2year,MET_2year_uncen,LR_2year,
                    LR_2year_uncen,RR_2year,RR_2year_uncen,LRR_2year,LRR_2year_uncen,DFS_2year,DFS_2year_uncen)
    pickle.dump(np.asarray(EndPoints), open('./EndPoints_clinical.d','wb'))    

    for i in range(0,606):

        #if i in ExcPT: 
            #continue

        filename_data = './VolPatch_clinical/VolPatch_'+ str(i) +'_data.d'

        # to extract imaging info
        input_info = AllPaRTFile[i] 
        #print (input_info[4].keys())
        try:
            PTSliceNo = sorted(input_info[4][obj2dis],key=int,reverse=False)

            dicominfo  = dicom.read_file(input_info[1][0])
            PixelSpacing = dicominfo.PixelSpacing
            SliceThickness = dicominfo.SliceThickness
            # if not SliceThickness:
            #     SliceThickness = 3

            CTResolution = [float(SliceThickness), float(PixelSpacing[0]), float(PixelSpacing[1])]
            pre_cropslice = int(np.ceil(crop_slice/(SliceThickness/resized_spacing[0])))
            pre_crop_sz = int(np.ceil(crop_sz/(PixelSpacing[0]/resized_spacing[1])))

            # CTimgs,CropSliceNo = GetCTSlices_2Drescale(input_info,PTSliceNo,actual_cropslice,CTResolution,resized_spacing)
            CTimgs,CropSliceNo = GetCTSlices(input_info,PTSliceNo,pre_cropslice)
            # CTimgs = CTimgs.astype(np.float)/np.max(CTimgs)

            MasksVol, MasksPT, SliceCentroid = GenerateMask(input_info,obj2dis,PTSliceNo,CropSliceNo)

            CroppedCTimgs = CropPTRegion(CTimgs,SliceCentroid,pre_crop_sz)
            CroppedMask = CropPTRegion(MasksVol,SliceCentroid,pre_crop_sz)

            RescaledCTimgs = RescaleVolumeData(CroppedCTimgs, CTResolution, resized_spacing)
            RescaledMasksVol = RescaleVolumeData(CroppedMask, CTResolution, resized_spacing)

            CroppedCTimgs = Crop3DCenterRegion(RescaledCTimgs,crop_sz,crop_slice)
            CroppedMask = Crop3DCenterRegion(RescaledMasksVol,crop_sz,crop_slice)

            CroppedMask[CroppedMask<0.5] = 0
            CroppedMask[CroppedMask>=0.5] = 1
    
            VolData = (CroppedCTimgs, CroppedMask, 
                    OS[i],TIME_OS[i],TSS[i],TIME_TSS[i],MET[i],TIME_MET[i],LR[i],TIME_LR[i],RR[i],TIME_RR[i],LRR[i],TIME_LRR[i],DFS[i],TIME_DFS[i],
                    OS_2year[i],OS_2year_uncen[i],TSS_2year[i],TSS_2year_uncen[i],MET_2year[i],MET_2year_uncen[i],LR_2year[i],LR_2year_uncen[i],RR_2year[i],RR_2year_uncen[i],LRR_2year[i],LRR_2year_uncen[i],DFS_2year[i],DFS_2year_uncen[i],
                    WHO_SCORE_codes_0VS123[i],Age[i],Gender[i],Smoking[i],TSTAD_DEF_codes_T123VS4[i],NSTAD_DEF_codes_N01VSN2VSN3[i],P16_codes[i]
                    )
            VolInfo = {'input_info':input_info[0], 'CropSliceNo':CropSliceNo}
                   
            # VolData = (CroppedCTimgs, CroppedMask,
            #         MET, TIMEMET, LocationSite, LRR, TIMELRR, RRR, TIMERRR, IndexTumorDEATH, TIMEITDEATH, DFS, TIMEDFS, OverallSurvival, TIMEOS,
            #         GenderVec, AgeFloat, ModalityCatVec, WHO_ScoreVec, TstageVec, NstageVec, ClinicalStageVec, TS_HPVposVec)
            '''
            VolData = {'CroppedCTimgs':CroppedCTimgs, 'CroppedMask':CroppedMask, 
                    'OS':OS[i],'TIME_OS':TIME_OS[i],'TSS':TSS[i],'TIME_TSS':TIME_TSS[i],'MET':MET[i],'TIME_MET':TIME_MET[i],'LR':LR[i],
                    'TIME_LR':TIME_LR[i],'RR':RR[i],'TIME_RR':TIME_RR[i],'LRR':LRR[i],'TIME_LRR':TIME_LRR[i],'DFS':DFS[i],'TIME_DFS':TIME_DFS[i],
                    'OS_2year':OS_2year[i],'OS_2year_uncen':OS_2year_uncen[i],'TSS_2year':TSS_2year[i],'TSS_2year_uncen':TSS_2year_uncen[i],'MET_2year':MET_2year[i],
                    'MET_2year_uncen':MET_2year_uncen[i],'LR_2year':LR_2year[i],'LR_2year_uncen':LR_2year_uncen[i],'RR_2year':RR_2year[i],'RR_2year_uncen':RR_2year_uncen[i],
                    'LRR_2year':LRR_2year[i],'LRR_2year_uncen':LRR_2year_uncen[i],'DFS_2year':DFS_2year[i],'DFS_2year_uncen':DFS_2year_uncen[i],'WHO_0VS123':WHO_0VS123[i],
                    'Age':Age[i],'Gender':Gender[i],'Smoking':Smoking[i],'TSTAD_T123VS4':TSTAD_T123VS4[i],'NSTAD_DEF_codes_N012VSN3':NSTAD_DEF_codes_N012VSN3[i],'P16_NegUnkVSPos':P16_NegUnkVSPos[i]
                    }
             
            
            VolData = pd.DataFrame(VolData)
            VolInfo = pd.DataFrame(VolInfo)
            
            VolData.to_csv('F:\\HN\\OPC-Radiomics\\VolPatch_clinical\\VolPatch_'+ str(i) +'_data.csv')
            VolInfo.to_csv('F:\\HN\\OPC-Radiomics\\VolPatch_clinical\\VolPatch_'+ str(i) +'_Info.csv')
            '''
            filename_data = './VolPatch_clinical/VolPatch_'+ str(i) +'_data.d'
            filename_info = './VolPatch_clinical/VolPatch_'+ str(i) +'_info.d'	

            pickle.dump(VolData,open(filename_data,'wb'))
            pickle.dump(VolInfo,open(filename_info,'wb'))	

            ValidDataInd[i] = i
        except:
            continue
            
    pickle.dump(ValidDataInd, open('./VolPatch_clinical/ValidDataInd_clinical.d','wb'))

def CategoricalPara(Para,nob=None):
    # this function convert number to one-hot coding
    if nob is None:
	    nobyte = max(Para) + 1 
    else:
        nobyte = nob
    
    vec = np.zeros(nobyte, dtype=int)
    if Para !=-1:
        vec[Para] = 1
        
    return vec

    
def Crop3DCenterRegion(Volume,crop_sz,crop_slice):
    # this function is used to crop a 3D volume

    VolCenter = np.asarray(Volume.shape)/2

    CroppedVolume = Volume[int(VolCenter[0])-crop_slice:int(VolCenter[0])+crop_slice,\
                            int(VolCenter[1])-crop_sz:int(VolCenter[1])+crop_sz,\
                            int(VolCenter[2])-crop_sz:int(VolCenter[2])+crop_sz]

    return CroppedVolume



def CropPTRegion(img, CropCenter, CropSZ, Masking=None):
	# generate cropped PT region in CT images
	# pdb.set_trace()

	CroppedImg = img[:,CropCenter[0]-CropSZ:CropCenter[0]+CropSZ,CropCenter[1]-CropSZ:CropCenter[1]+CropSZ]
	if Masking is not None: # the masked img
		Masking = Masking[:,CropCenter[0]-CropSZ:CropCenter[0]+CropSZ,CropCenter[1]-CropSZ:CropCenter[1]+CropSZ]
		CroppedImg = CroppedImg*Masking

	return CroppedImg



def GetAllCTSlices(input_info):
    IndSlice = input_info[3]
    sample_paths = [input_info[1][ind] for ind in IndSlice]
    CTimgs = [dicom.read_file(sample_path).pixel_array for sample_path in sample_paths]
    dicominfo  = dicom.read_file(sample_paths[0])
    PixelSpacing = dicominfo.PixelSpacing
    SliceThickness = dicominfo.SliceThickness
    CTimgs = np.asarray(CTimgs)
    Resolution = [float(PixelSpacing[0]), float(PixelSpacing[1]), float(SliceThickness)]
    
    return CTimgs,Resolution


def RescaleVolumeData(InputVolume,OriResolution,RescaledResolution):
    
    # this function is used to rescale the volumetric data to a certain resolution
    resize_factor = np.asarray(OriResolution)/np.asarray(RescaledResolution)
    # RescaledSize = np.round(InputVolume.shape*resize_factor)


    RescaledVolume = ndimage.interpolation.zoom(InputVolume, resize_factor)

    # Rescaled_xy = []
    # Rescaled_z = []


    # # recale along x-y axis
    # for i in range(InputVolume.shape[0]):
    #     RescaledSlice = ndimage.interpolation.zoom(InputVolume[i], RescaledSize[1:])
    #     Rescaled_xy.append(RescaledSlice)    

    # Rescaled_xy = np.asarray(Rescaled_xy)

    # # recale along z axis
    # for i in range(InputVolume.shape[1]):
    #     RescaledZ = ndimage.interpolation.zoom(Rescaled_xy[:,:,i], RescaledSize[:2])
    #     Rescaled_z.append(RescaledZ)

    # RescaledVolume = Rescaled_z

    return RescaledVolume




def GetCTSlices(input_info,PTsliceNo,crop_slice):
	# Get the CT slices with the minimum required number of slices
    NoSlice = len(input_info[1])
    CropSliceNo = range(int(PTsliceNo[int(len(PTsliceNo)/2)])-crop_slice,int(PTsliceNo[int(len(PTsliceNo)/2)])+crop_slice)
	
    if CropSliceNo[0]<0:
        CropSliceNo = range(0,CropSliceNo[-1]+abs(CropSliceNo[0]))
    elif CropSliceNo[-1]>NoSlice:
        CropSliceNo = range(CropSliceNo[0]-(CropSliceNo[-1]-NoSlice),NoSlice)

    IndSlice = input_info[3] 

    sample_paths = [input_info[1][ind] for ind in IndSlice[np.asarray(CropSliceNo).astype(int)]]

    CTimgs = [dicom.read_file(sample_path).pixel_array for sample_path in sample_paths]
    CTimgs = np.asarray(CTimgs)
	
    # change to HU value
    CT_header = dicom.read_file(sample_paths[0])
    RescaleIntercept = CT_header.RescaleIntercept
    RescaleSlope = CT_header.RescaleSlope
    #print ('slope, intercept',  RescaleSlope, RescaleIntercept)
    #print ('CT max, CTmin', CTimgs.max(), CTimgs.min())
    # now get real HU for opc-Radiomics
    CTimgs = CTimgs* RescaleSlope + RescaleIntercept
	
    return CTimgs,CropSliceNo
	
def GenerateMask(input_info,obj2dis,PTsliceNo,CropSliceNo):
	# generate binary masks from the PT region coordinates
	
	ROIContours = input_info[4]
	MasksVol = np.zeros((len(CropSliceNo),512,512))
	MasksPT = []
	Centroid = []

	for i_slice in range(0,len(PTsliceNo)):
		
		SliceNum = PTsliceNo[i_slice]
		cnmask = Image.new('L',(512,512),0)
		for i_region in range(0,len(ROIContours[obj2dis][SliceNum])):

			assert np.array(int(SliceNum)) == np.unique(ROIContours[obj2dis][SliceNum][i_region][:,2])
			polygonpoints = np.reshape(ROIContours[obj2dis][SliceNum][i_region][:,0:2],(len(ROIContours[obj2dis][SliceNum][i_region][:,0:2])*2))
			
			ImageDraw.Draw(cnmask).polygon(polygonpoints.tolist(),outline=1,fill=1)
			
		props = skimage.measure.regionprops(np.array(cnmask))
		
		MasksPT.append(cnmask)
		Centroid.append([props[0].centroid[0],props[0].centroid[1]])
		# pdb.set_trace()

		# try:
		MasksVol[int(SliceNum)-CropSliceNo[0],:,:] = np.array(cnmask)
		# except:
		# 	pdb.set_trace()			

	center = np.average(Centroid,axis=0)
	SliceCentroid = (int(center[0]),int(center[1]))

	return MasksVol, MasksPT, SliceCentroid

if __name__ == '__main__':
	
	main()
