#This is the official code of paper "Comparison of CT image features extracted by radiomics, self-supervised learning and end-to-end deep learning for outcome prediction of oropharyngeal cancer.(2023, Physics and Imaging in Radiation Oncology, under review)"

# Data preprocessing:
cd Data_preprocessing

python OriFile2Digit_opcradiomics.py # clinical and outcome preprocessing

python org_OPCRadiomics_3Dpatch.py # select patients who have GTV

python Crop3DPatches_ClinialCategory_OPCradiomics.py # save clinical and crop image data into pickle file


#To train the autoencoder, use   python  PTmain_3DFeatureExtr.py"  --input_type 3  --input_modality 0  --result_path ...  --data_path ... 

#To test the autoencoder, use   python  PTmain_3DFeatureExtr.py"  --input_type 3  --input_modality 0  --result_path ...  --data_path ... --no_train --no_val --test --resume_path save_80.pth
