__project__   = "DAME"
__author__    = "ZKH\guoj"
__function__  = "This function is used to extract rtstruct info in the OPSCC dataset for CT-based prognostic outcome prediction" 
__version__   = "0.0.1"

import os
import glob
import pydicom as dicom
import numpy as np
from PIL import Image,ImageDraw
import skimage.measure 
import matplotlib.pyplot as plt
import matplotlib.image as pli
import pandas as pd


DATA_PATH = 'F:\\HN\\OPC-Radiomics\\OPC-Radiomics\\'

def main():

	#structname = 'GTV' #''GTVpt'CTV70','CTV50','PTV70'
	AllPaRTFile = []

	PTno = 0
    
	#list_ctv = pd.read_excel('structurelist-GTV.xlsx')
	#list_ctv = list(list_ctv['No GTV'])
	#print (list_ctv)

	for pt in range(1,607):
      
		# OPCRadiomics

		# 228 --> there's GTVp but no contours
		# 270,279,295 --> all files are missing
		# 271 --> only ln are there

		# 217,311,355,361,454,456,461,476, 499, 550 --> no rtstruct
		# 226 --> GTVp1, GTVp2
		# 225,226,577,607 --> SliceNo has two values for one slice
		structname = 'GTV'        
		try:
			CTFiles, RTSTRUCTFiles = GetRTdcmNames_OPCRadiomics(DATA_PATH,pt)
			IndSlice, ROIContours, MaxSizeObj, StudyDate = GetROIAttributes_OPCRadiomics(CTFiles, RTSTRUCTFiles, structkeyword=structname)
			DispScanContour_HNPETCT(CTFiles,ROIContours,IndSlice,obj2dis=structname,saveIm=str(pt))

			obj2dis = structname
			if obj2dis in MaxSizeObj.keys():
				MaxPTDiameter = np.max(MaxSizeObj[obj2dis])
				NumPTSlice = len(ROIContours[obj2dis].keys())
			else:
				MaxPTDiameter = 0

			PaRTFile = ('OPC-'+'{0:05}'.format(pt), CTFiles, RTSTRUCTFiles, IndSlice, ROIContours, MaxPTDiameter, NumPTSlice, StudyDate)
			AllPaRTFile.append(PaRTFile)
			print('Finished: Patient No. %d with %s' % (pt,pt))
			PTno += 1

		except:
			try:
				structname = 'CTV70'
				CTFiles, RTSTRUCTFiles = GetRTdcmNames_OPCRadiomics(DATA_PATH,pt)
				IndSlice, ROIContours, MaxSizeObj, StudyDate = GetROIAttributes_OPCRadiomics(CTFiles, RTSTRUCTFiles, structkeyword=structname)
				DispScanContour_HNPETCT(CTFiles,ROIContours,IndSlice,obj2dis=structname,saveIm=str(pt))

				obj2dis = structname
				if obj2dis in MaxSizeObj.keys():
					MaxPTDiameter = np.max(MaxSizeObj[obj2dis])
					NumPTSlice = len(ROIContours[obj2dis].keys())
				else:
					MaxPTDiameter = 0
				PaRTFile = ('OPC-'+'{0:05}'.format(pt), CTFiles, RTSTRUCTFiles, IndSlice, ROIContours, MaxPTDiameter, NumPTSlice, StudyDate)
				AllPaRTFile.append(PaRTFile)
				print('Finished: Patient No. %d with %s' % (pt,pt))
				PTno += 1
			except:
				try:
					structname = 'CTV60'
					CTFiles, RTSTRUCTFiles = GetRTdcmNames_OPCRadiomics(DATA_PATH,pt)
					IndSlice, ROIContours, MaxSizeObj, StudyDate = GetROIAttributes_OPCRadiomics(CTFiles, RTSTRUCTFiles, structkeyword=structname)
					DispScanContour_HNPETCT(CTFiles,ROIContours,IndSlice,obj2dis=structname,saveIm=str(pt))

					obj2dis = structname
					if obj2dis in MaxSizeObj.keys():
					  MaxPTDiameter = np.max(MaxSizeObj[obj2dis])
					  NumPTSlice = len(ROIContours[obj2dis].keys())
					else:
					  MaxPTDiameter = 0
					PaRTFile = ('OPC-'+'{0:05}'.format(pt), CTFiles, RTSTRUCTFiles, IndSlice, ROIContours, MaxPTDiameter, NumPTSlice, StudyDate)
					AllPaRTFile.append(PaRTFile)
					print('Finished: Patient No. %d with %s' % (pt,pt))
					PTno += 1
                
				except:
					try:
					  structname = 'CTV50'
					  CTFiles, RTSTRUCTFiles = GetRTdcmNames_OPCRadiomics(DATA_PATH,pt)
					  IndSlice, ROIContours, MaxSizeObj, StudyDate = GetROIAttributes_OPCRadiomics(CTFiles, RTSTRUCTFiles, structkeyword=structname)
					  DispScanContour_HNPETCT(CTFiles,ROIContours,IndSlice,obj2dis=structname,saveIm=str(pt))

					  obj2dis = structname
					  if obj2dis in MaxSizeObj.keys():
					    MaxPTDiameter = np.max(MaxSizeObj[obj2dis])
					    NumPTSlice = len(ROIContours[obj2dis].keys())
					  else:
					    MaxPTDiameter = 0
					  PaRTFile = ('OPC-'+'{0:05}'.format(pt), CTFiles, RTSTRUCTFiles, IndSlice, ROIContours, MaxPTDiameter, NumPTSlice, StudyDate)
					  AllPaRTFile.append(PaRTFile)
					  print('Finished: Patient No. %d with %s' % (pt,pt))
					  PTno += 1
					except:
					  try:
					    structname = 'CTV63'
					    CTFiles, RTSTRUCTFiles = GetRTdcmNames_OPCRadiomics(DATA_PATH,pt)
					    IndSlice, ROIContours, MaxSizeObj, StudyDate = GetROIAttributes_OPCRadiomics(CTFiles, RTSTRUCTFiles, structkeyword=structname)
					    DispScanContour_HNPETCT(CTFiles,ROIContours,IndSlice,obj2dis=structname,saveIm=str(pt))

					    obj2dis = structname
					    if obj2dis in MaxSizeObj.keys():
					        MaxPTDiameter = np.max(MaxSizeObj[obj2dis])
					        NumPTSlice = len(ROIContours[obj2dis].keys())
					    else:
					        MaxPTDiameter = 0
					    PaRTFile = ('OPC-'+'{0:05}'.format(pt), CTFiles, RTSTRUCTFiles, IndSlice, ROIContours, MaxPTDiameter, NumPTSlice, StudyDate)
					    AllPaRTFile.append(PaRTFile)
					    print('Finished: Patient No. %d with %s' % (pt,pt))                        
					    PTno += 1
					  except:
					    PaRTFile = ('OPC-'+'{0:05}'.format(pt),)
					    AllPaRTFile.append(PaRTFile)
					    print('Finished_prob: Patient No. %d with %s' % (pt,pt))
					    PTno += 1
					    continue


	import pickle
	pickle.dump(AllPaRTFile,open('AllScanROIFile_OPCRadiomics.d','wb'))	


def GetRTdcmNames_OPCRadiomics(Data_path,PaID):
# this function reads all CT and RTSTRUCT files names in OPCRadiomics

	CTfilename = []
	RTSTRUCTfilename = []

	PtFolder = os.path.join(Data_path+'OPC-'+'{0:05}'.format(PaID))
	PtSubfolders = next(os.walk(PtFolder))[1]
	ScanStructFolders = next(os.walk(PtFolder+"\\"+PtSubfolders[0]))[1]

	for i in range(0,len(ScanStructFolders)):
		FilesDir = os.path.join(PtFolder,PtSubfolders[0],ScanStructFolders[i]) + "\\*.dcm"
		Files = sorted(glob.glob(FilesDir)) # sort the files by name

		if len(Files)==1:
			DcmFile = dicom.read_file(Files[0])
			if DcmFile.Modality == 'RTSTRUCT':
				RTSTRUCTfilename.append(Files[0])
		else:
			CTfilename.extend(Files)	
    
	return CTfilename, RTSTRUCTfilename

def GetROIAttributes_OPCRadiomics(CTFilesnames, STRUCTFilenames, structkeyword='GTV'):

	#-----------------------------------------------------------------------------
	# get SOPInstanceUID, Image Position Patient, Pixel Spacing from the CT dicoms
	# SOPInstanceUID is used to order the CT slices
	#-----------------------------------------------------------------------------
	
	# pdb.set_trace()

	ImagePositionPatient = []
	PixelSpacing = []
	SOPInstanceUID = []

	if structkeyword == 'GTV':
		keywordlist = ['GTV']
	elif structkeyword == 'CTV70':
		keywordlist = ['CTV70']
	elif structkeyword == 'CTV50':
		keywordlist = ['CTV50']
	elif structkeyword == 'CTV60':
		keywordlist = ['CTV60']
	elif structkeyword == 'CTV63':
		keywordlist = ['CTV63']
	elif structkeyword == 'PTV70':
		keywordlist = ['PTV70']

	for i in range(0,len(CTFilesnames)):

		dicomheader = dicom.read_file(CTFilesnames[i])
		SOPInstanceUID_perdcm = np.array(dicomheader.SOPInstanceUID[-4:]) # the last four digits


		PosiPatient = np.asarray(dicomheader.ImagePositionPatient)
		PixelSp = np.asarray(dicomheader.PixelSpacing)

		ImagePositionPatient.append(PosiPatient) 
		PixelSpacing.append(PixelSp) 
		SOPInstanceUID.append(SOPInstanceUID_perdcm)


	ImagePositionPatient = np.asarray(ImagePositionPatient).astype(np.float)
	PixelSpacing = np.asarray(PixelSpacing).astype(np.float)

	SOPInstanceUID = np.asarray(SOPInstanceUID)
	IndSlice = np.argsort(ImagePositionPatient[:,2]) 
	# the indices to order the slices of CT scans, SOPInstanceUID is not used since the order can be desending


	Z_ImagePositionPatient = ImagePositionPatient[IndSlice,2]
	ImagePositionPatient = ImagePositionPatient[IndSlice,:] 
	PositionOffset = ImagePositionPatient[0,:]

	slice_space=(np.max(Z_ImagePositionPatient)-np.min(Z_ImagePositionPatient))/(len(Z_ImagePositionPatient)-1) # actually, it's the same as slice thickness

	#----------------------------------------------------------------------------------------
	# get Contour points
	# the data points are saved as a one dimensional vector in the order of [x,y,z,x,y,z....]
	#----------------------------------------------------------------------------------------
	# pdb.set_trace()

	# for i_series in range(0,len(STRUCTFilenames)): 
	dicominfotemp = dicom.read_file(STRUCTFilenames[0])
	RTSTRUCTdicomfile = dicominfotemp
	StudyDate = RTSTRUCTdicomfile.StudyDate
	
		# if 'disconnect' in dicominfotemp.SeriesDescription or 'disconnet' in dicominfotemp.SeriesDescription:
		# 	RTSTRUCTdicomfile = dicominfotemp
		# 	StudyDate = RTSTRUCTdicomfile.StudyDate
	
	ROIContours = {}
	MaxSizeObj = {}

	for i_roi in range(0,len(RTSTRUCTdicomfile.StructureSetROISequence)):

		# for i in range(0,len(RTSTRUCTdicomfile.StructureSetROISequence)):
		# 	print(i,RTSTRUCTdicomfile.StructureSetROISequence[i].ROIName)
		# pdb.set_trace()

		# List of ROI Name: CTV 57, CTV 57 Sub, PTV 57, CTV 60, CTV 60 Sub, PTV 60, CTV 70, CTV 70 Sub, PTV 70, 
		# GTV, GTV Nodes, Rt Parotid, Lt Parotid, Mandible

		if RTSTRUCTdicomfile.StructureSetROISequence[i_roi].ROIName in keywordlist: #['GTV', 'GTV Nodes', 'Mandible']: 
			try: 
				NumberOfContourSlices = len(RTSTRUCTdicomfile.ROIContourSequence[i_roi].ContourSequence)
			except:
				continue

			AllContourData = []
			AllNumberOfContourPoints = []
			AllContourCoord = []
			CNCoord = {}
			MaxSize = []
			# pdb.set_trace()

			for i_ctslice in range(0,NumberOfContourSlices):

				try:
					NumberOfContourPoints_perslice = np.asarray(RTSTRUCTdicomfile.ROIContourSequence[i_roi].ContourSequence[i_ctslice].NumberOfContourPoints)
				except:
					continue	

				if NumberOfContourPoints_perslice>2:
					ContourData_perslice = RTSTRUCTdicomfile.ROIContourSequence[i_roi].ContourSequence[i_ctslice].ContourData

					ContourData_perslice = np.asarray(ContourData_perslice)
					
					#------------------------------------------------------------
					# Correct the Contour points with the Image Position Patient
					#------------------------------------------------------------

					# % Picking out xyz coordinates from each individual contourpoint from
					# % variable 'structure data' and save them in variables x, y, and z for x, y,
					# % and z coordinates respectively. x, y, and z have to be altered for
					# % matching the coordinate systems of the CT images and the RT STRUCT.
					# % ImagePositionPatient, which describes the minimum value of the CT
					# % images in one direction, is subtracted from v1. This answer is
					# % divided by the distance of 2 different voxels among the corresponding
					# % axis. Eventually Matlabs function 'poly2mask', which computes a
					# % binary mask from the contour, is used to sum each contour.

					#ImagePositionPatient is in the original CT scan

					# three coordinates: x,y,z 
					x = ContourData_perslice[range(0,len(ContourData_perslice),3)]
					y = ContourData_perslice[range(1,len(ContourData_perslice),3)]
					z = ContourData_perslice[range(2,len(ContourData_perslice),3)]
					

					ContourDataPoints_perslice = np.array([x,y,z]).T
					ContourCoord_perslice = (ContourDataPoints_perslice-PositionOffset)/np.array([PixelSpacing[0,0],PixelSpacing[0,1],slice_space])
					ContourCoord_perslice = np.round(ContourCoord_perslice).astype(int)
					SliceNo = np.unique(ContourCoord_perslice[:,2])
					# if len(SliceNo)==2:
					# 	ContourCoord_perslice = (ContourDataPoints_perslice-PositionOffset)/np.array([PixelSpacing[0,0],PixelSpacing[0,1],slice_space])
					# 	ContourCoord_perslice = ContourCoord_perslice.astype(int)
					# 	SliceNo = np.unique(ContourCoord_perslice[:,2])

					assert len(SliceNo)==1 

					AllNumberOfContourPoints.append(NumberOfContourPoints_perslice)
					AllContourData.extend(ContourData_perslice)
					AllContourCoord.extend(ContourCoord_perslice)

					polygonpoints = np.reshape(ContourCoord_perslice[:,0:2],(NumberOfContourPoints_perslice*2))

					temp_mask = Image.new('L',(512,512),0)

					ImageDraw.Draw(temp_mask).polygon(polygonpoints.tolist(),outline=1,fill=1)
					
					props = skimage.measure.regionprops(np.array(temp_mask))
					if len(props)==0: # when there's no region
						continue
					maxbbox = max(props[0].bbox[2]-props[0].bbox[0],props[0].bbox[3]-props[0].bbox[1])

					MaxSize.append(maxbbox)

					if props[0].area>10: # we consider the small regions as noise points

						if str(SliceNo[0]) in CNCoord:
							CNCoord[str(SliceNo[0])].append(ContourCoord_perslice)
						else:
							CNCoord[str(SliceNo[0])] = [ContourCoord_perslice]
					else:
						z = 0
				

			ROIContours[structkeyword] = CNCoord
			MaxSizeObj[structkeyword] = MaxSize
			
			# break # once a ROISequence is found, the loop breaks
		
	# pdb.set_trace()

	return IndSlice, ROIContours, MaxSizeObj, StudyDate
	# return StudyDate



def DispScanContour_HNPETCT(CTDicom,ROIContours,IndSlice,obj2dis='GTV',saveIm=None):
	#---------------------------------------------------------------------------------------------------------------------
	# to display the Scan and Contours of ROI 
	# ROIContours is a dict that has all contour points of the ROIs. It is obtained from the def 'GetROIAttributes'
	# obj2dis: the object is going to be displayed, 'GTVpt'---> primary tumors/'GTVln'---> lymph node
	# IndSlice: the slice order
	# saveIm: save the display as .png or not 
	#---------------------------------------------------------------------------------------------------------------------


	PTsliceNo = sorted(ROIContours[obj2dis],key=int,reverse=False)

	for i_slice in range(0,len(PTsliceNo)):
		

		SliceNum = PTsliceNo[i_slice]
		cnmask = Image.new('L',(512,512),0)
		for i_region in range(0,len(ROIContours[obj2dis][SliceNum])):

			assert np.array(int(SliceNum)) == np.unique(ROIContours[obj2dis][SliceNum][i_region][:,2])
			polygonpoints = np.reshape(ROIContours[obj2dis][SliceNum][i_region][:,0:2],(len(ROIContours[obj2dis][SliceNum][i_region][:,0:2])*2))
			
			ImageDraw.Draw(cnmask).polygon(polygonpoints.tolist(),outline=1,fill=1)

		slice_header = dicom.read_file(CTDicom[IndSlice[int(SliceNum)]])

		scan = slice_header.pixel_array

		# apply head and neck windowing to the CT slices
		# Window width W = 300, Window level L = 40
	    # HU = grayintensity*slope+intercept
		# The gray intensity lower and upper thresholds: 
		# LowTh = ((L-W/2)-intercept)/slope, UpperTh = ((L+W/2)-intercept)/slope 

		W = 300
		L = 40
		RescaleIntercept = slice_header.RescaleIntercept
		RescaleSlope = slice_header.RescaleSlope

		# convert the ones that stored as the 'HU values' to intensity values
		if scan.min()<0:
			RescaleIntercept = scan.min()
			scan += abs(scan.min())

		LowTh = ((L-W/2)-RescaleIntercept)/RescaleSlope
		UpperTh = ((L+W/2)-RescaleIntercept)/RescaleSlope

		scan[scan<LowTh] = LowTh
		scan[scan>UpperTh] = UpperTh

		scan_im = (scan.astype(np.float)-LowTh)/np.max(scan)
		scan_im = Image.fromarray(scan_im)

		scan_im_colorlay_pt = np.zeros((512,512,3))

		
		scan_im_colorlay_pt[:,:,0] = scan_im + np.asarray(cnmask)*0.4
		scan_im_colorlay_pt[:,:,1] = scan_im
		scan_im_colorlay_pt[:,:,2] = scan_im 
		'''
		plt.figure()
		plt.subplot(1,3,1)
		plt.imshow(cnmask)
		plt.subplot(1,3,2)
		plt.imshow(scan,cmap=plt.cm.gray)
		plt.subplot(1,3,3)
		plt.imshow(scan_im_colorlay_pt)
		plt.title('Slice No. %s'%(int(SliceNum)+1))
		plt.show()
		'''
		# pdb.set_trace()

		if saveIm != None:

			if not os.path.exists('F:\\HN\\OPC-Radiomics\\%s\\'%(obj2dis)+saveIm):
				os.mkdir('F:\\HN\\OPC-Radiomics\\%s\\'%(obj2dis)+saveIm)
				
			scanname = 'F:\\HN\\OPC-Radiomics\\%s\\%s\\Scan_SliceNo%s.png'%(obj2dis,saveIm,int(SliceNum)+1)
			maskname = 'F:\\HN\\OPC-Radiomics\\%s\\%s\\%sMask_SliceNo%s.png'%(obj2dis,saveIm,obj2dis,int(SliceNum)+1)
			scanname_colorlay_pt = 'F:\\HN\\OPC-Radiomics\\%s\\%s\\%sScan_colorlay_SliceNo%s.png'%(obj2dis,saveIm,obj2dis,int(SliceNum)+1)
			pli.imsave(scanname,scan_im,cmap=plt.cm.gray)
			pli.imsave(maskname,cnmask,cmap=plt.cm.gray)
			pli.imsave(scanname_colorlay_pt,scan_im_colorlay_pt)



if __name__ == '__main__':
	main()