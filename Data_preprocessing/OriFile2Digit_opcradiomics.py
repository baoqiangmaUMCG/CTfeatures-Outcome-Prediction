'''
The code of process clinical data and outcome 
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
#import seaborn as sns
#sns.set(style="white")
#sns.set(style="whitegrid", color_codes=True)



opcradiomics = pd.ExcelFile('./Clinical data list_OPC (606) v3_newcolumnname.xlsx').parse(0) # clinical data
opcradiomics.dropna()

#-----preparation of the opcradiomics data
#----clinical parameters
opcradiomics['GESLACHT_codes']    = 1-opcradiomics['GESLACHT'].astype('category').cat.codes #0-Man,1-Vrouw

opcradiomics['Smoking_codes']     = opcradiomics['Smoking'].astype('category').cat.codes.copy() # 0-Current, 1-Never, 2-Past

opcradiomics['Smoking_codes'].loc[np.where(opcradiomics['Smoking']=='Ex-smoker')] = 1 #change to 0-Current, 2-Never, 1-Past
opcradiomics['Smoking_codes'].loc[np.where(opcradiomics['Smoking']=='Current')] = 0
opcradiomics['Smoking_codes'].loc[np.where(opcradiomics['Smoking']=='Non-smoker')] = 2
# new 
opcradiomics['Smoking_codes_noVSyes'] = opcradiomics['Smoking_codes']
opcradiomics['Smoking_codes_noVSyes'] = np.where(opcradiomics['Smoking_codes']==2,0,1) # 0-never smoke, 1-current or ever smoke

opcradiomics['MODALITY_codes']    = 1-opcradiomics['MODALITY (chemo/not)'].astype('category').cat.codes # 0-RT only 1-chemo 

opcradiomics['TSTAD_codes']   = opcradiomics['TSTAD_DEF'].astype('category').cat.codes # 0-T1,1-T2,2-T3,3-T4a,4-T4b
opcradiomics['TSTAD_codes'].loc[np.where(opcradiomics['TSTAD_DEF']=='T4b')] = 3  # 0-T1,1-T2,2-T3,3-T4


# new
opcradiomics['TSTAD_codes_12VS34'] = opcradiomics['TSTAD_codes']
opcradiomics['TSTAD_codes_12VS34'].loc[np.where(opcradiomics['TSTAD_codes']<2)] = 0  # 0-T1T2
opcradiomics['TSTAD_codes_12VS34'].loc[np.where(opcradiomics['TSTAD_codes']>1)] = 1  # 0-T3T4

opcradiomics['NSTAD_codes']   = opcradiomics['NSTAD_DEF'].astype('category').cat.codes # 0-N0,1-N1,2-N2a,3-N2b,4-N2c,5-N3

# new
opcradiomics['NSTAD_codes_012VS3'] = opcradiomics['NSTAD_codes']
opcradiomics['NSTAD_codes_012VS3'].loc[np.where(opcradiomics['NSTAD_codes']<5)] = 0  # 0-N0N1N2
opcradiomics['NSTAD_codes_012VS3'].loc[np.where(opcradiomics['NSTAD_codes']==5)] = 1  # 0-N3

# new
opcradiomics['P16_codes']  = opcradiomics['P16'].astype('category').cat.codes # 0-negative,1-positive,2-unknown
opcradiomics['P16_codes_combine'] = np.where(opcradiomics['P16_codes']==0,0,1) # 0-negative , 1-positive and unknown

opcradiomics['WHO_SCORE_codes']   = opcradiomics['WHO_SCORE'].astype('category').cat.codes # 0- ecog 0, 1-ecog 1, 2-ecog 2, 3-ecog 3, 4-ecog4,-1-non
opcradiomics['WHO_SCORE_codes'].loc[np.where(opcradiomics['WHO_SCORE_codes']==-1)] = 5

# old
opcradiomics['TSTAD_codes_123VS4'] = np.where(opcradiomics['TSTAD_codes']==3,1,0)
opcradiomics['NSTAD_codes_N01VSN2VSN3'] = opcradiomics['NSTAD_codes']
opcradiomics['NSTAD_codes_N01VSN2VSN3'].loc[np.where(opcradiomics['NSTAD_codes_N01VSN2VSN3']==1)] = 0
opcradiomics['NSTAD_codes_N01VSN2VSN3'].loc[np.where(opcradiomics['NSTAD_codes_N01VSN2VSN3']==2)] = 1
opcradiomics['NSTAD_codes_N01VSN2VSN3'].loc[np.where(opcradiomics['NSTAD_codes_N01VSN2VSN3']==3)] = 1
opcradiomics['NSTAD_codes_N01VSN2VSN3'].loc[np.where(opcradiomics['NSTAD_codes_N01VSN2VSN3']==4)] = 1
opcradiomics['NSTAD_codes_N01VSN2VSN3'].loc[np.where(opcradiomics['NSTAD_codes_N01VSN2VSN3']==5)] = 2

opcradiomics['WHO_SCORE_codes_0VS123'] = np.where(opcradiomics['WHO_SCORE_codes']==0,0,1)
'''
numeric_columns = ['AGE']
opcdata[numeric_columns] = opcdata[numeric_columns]/100.
'''  
#-----endpoints

opcradiomics['Status(OS)'].value_counts()
opcradiomics['STATUS_code'] = opcradiomics['Status(OS)'].astype('category').cat.codes  # 0-alive,1-dead

opcradiomics['OS_code'] = opcradiomics['STATUS_code'] # make a copy
opcradiomics['TIME_OS'] = (opcradiomics['Time interval from the date of diagnosis to the date of last FU  (days)']-opcradiomics['Time interval from the date of diagnosis to the RT start date (days)'])/30.0

opcradiomics['OS_2year'] = opcradiomics['OS_code']
opcradiomics['OS_2year'].loc[list(opcradiomics.loc[(opcradiomics['OS_code']==1) & (opcradiomics['TIME_OS']>24)].index)]=0 # -100 
opcradiomics['OS_2year_uncensoring'] = 1
opcradiomics['OS_2year_uncensoring'].loc[list(opcradiomics.loc[(opcradiomics['OS_code']==0) & (opcradiomics['TIME_OS']<24)].index)]=0 #


opcradiomics['TumorSpecificSurvival'] = opcradiomics['Cause of Death'] # make a copy
opcradiomics['TumorSpecificSurvival_code'] = np.where(opcradiomics['TumorSpecificSurvival']=='Index Cancer',1,0)
opcradiomics['TIME_TumorSpecificSurvival'] = opcradiomics['TIME_OS'] # tumor specific survival

opcradiomics['TumorSpecificSurvival_2year'] = opcradiomics['TumorSpecificSurvival_code']
opcradiomics['TumorSpecificSurvival_2year'].loc[list(opcradiomics.loc[(opcradiomics['TumorSpecificSurvival_code']==1) & (opcradiomics['TIME_TumorSpecificSurvival']>24)].index)]=0 # -35
opcradiomics['TumorSpecificSurvival_2year_uncensoring'] = 1
opcradiomics['TumorSpecificSurvival_2year_uncensoring'].loc[list(opcradiomics.loc[(opcradiomics['TumorSpecificSurvival_code']==0) & (opcradiomics['TIME_TumorSpecificSurvival']<24)].index)]=0 #

opcradiomics['LR_code']=np.where(opcradiomics['Local Failure']=='Yes',1,0) # 0-no LR, 1- yes LR
opcradiomics['TIME_LR'] = (opcradiomics['TIME_diagnosis2LF (days)']-opcradiomics['Time interval from the date of diagnosis to the RT start date (days)'])/30.0

opcradiomics['LR_code_2year'] = opcradiomics['LR_code']
opcradiomics['LR_code_2year'].loc[list(opcradiomics.loc[(opcradiomics['LR_code']==1) & (opcradiomics['TIME_LR']>24)].index)]=0 # -4
opcradiomics['LR_code_2year_uncensoring'] = 1
opcradiomics['LR_code_2year_uncensoring'].loc[list(opcradiomics.loc[(opcradiomics['LR_code']==0) & (opcradiomics['TIME_LR']<24)].index)]=0 #


opcradiomics['RR_code']=np.where(opcradiomics['Regional Failure']=='Yes',1,0) # 0-no LR, 1- yes LR
opcradiomics['TIME_RR'] = (opcradiomics['TIME_diagnosis2RF (days)']-opcradiomics['Time interval from the date of diagnosis to the RT start date (days)'])/30.0

opcradiomics['RR_code_2year'] = opcradiomics['RR_code']
opcradiomics['RR_code_2year'].loc[list(opcradiomics.loc[(opcradiomics['RR_code']==1) & (opcradiomics['TIME_RR']>24)].index)]=0 # -267
opcradiomics['RR_code_2year_uncensoring'] = 1
opcradiomics['RR_code_2year_uncensoring'].loc[list(opcradiomics.loc[(opcradiomics['RR_code']==0) & (opcradiomics['TIME_RR']<24)].index)]=0 #



opcradiomics['LRR_code'] = opcradiomics['LR_code']+opcradiomics['RR_code']
opcradiomics['LRR_code'] = np.where(opcradiomics['LRR_code']==0,0,1)
opcradiomics['TIME_LRR'] = np.minimum(opcradiomics['TIME_LR'],opcradiomics['TIME_RR'])

opcradiomics['LRR_code_2year'] = opcradiomics['LRR_code']
opcradiomics['LRR_code_2year'].loc[list(opcradiomics.loc[(opcradiomics['LRR_code']==1) & (opcradiomics['TIME_LRR']>24)].index)]=0 # -7
opcradiomics['LRR_code_2year_uncensoring'] = 1
opcradiomics['LRR_code_2year_uncensoring'].loc[list(opcradiomics.loc[(opcradiomics['LRR_code']==0) & (opcradiomics['TIME_LRR']<24)].index)]=0 # 



opcradiomics['MET_code'] = np.where(opcradiomics['Distant Failure']=='Yes',1,0) # 0-no LR, 1- yes LR
opcradiomics['TIME_MET'] = (opcradiomics['TIME_diagnosis2DF (days)']-opcradiomics['Time interval from the date of diagnosis to the RT start date (days)'])/30.0

opcradiomics['MET_code_2year'] = opcradiomics['MET_code'] 
opcradiomics['MET_code_2year'].loc[list(opcradiomics.loc[(opcradiomics['MET_code']==1) & (opcradiomics['TIME_MET']>24)].index)]=0 # -8 correct
opcradiomics['MET_code_2year_uncensoring'] = 1
opcradiomics['MET_code_2year_uncensoring'].loc[list(opcradiomics.loc[(opcradiomics['MET_code']==0) & (opcradiomics['TIME_MET']<24)].index)]=0 # 



opcradiomics['DFS_code'] = opcradiomics['LR_code']+opcradiomics['RR_code']+opcradiomics['MET_code']+opcradiomics['OS_code']
opcradiomics['DFS_code'] = np.where(opcradiomics['DFS_code']==0,0,1)
opcradiomics['TIME_DFS'] = np.minimum(np.minimum(opcradiomics['TIME_MET'],opcradiomics['TIME_OS']),opcradiomics['TIME_LRR'])

opcradiomics['DFS_code_2year'] = opcradiomics['DFS_code']
opcradiomics['DFS_code_2year'].loc[list(opcradiomics.loc[(opcradiomics['DFS_code']==1) & (opcradiomics['TIME_DFS']>24)].index)]=0 # -254
opcradiomics['DFS_code_2year_uncensoring'] = 1
opcradiomics['DFS_code_2year_uncensoring'].loc[list(opcradiomics.loc[(opcradiomics['DFS_code']==0) & (opcradiomics['TIME_DFS']<24)].index)]=0 #




clinical_para = ['AGE','GESLACHT_codes','Smoking_codes','Smoking_codes_noVSyes','MODALITY_codes','TSTAD_codes','TSTAD_codes_123VS4','TSTAD_codes_12VS34', 'NSTAD_codes','NSTAD_codes_N01VSN2VSN3','NSTAD_codes_012VS3', 
                 'P16_codes', 'P16_codes_combine', 'WHO_SCORE_codes','WHO_SCORE_codes_0VS123']

# endpoint (original and correct for two year prediction) and survival

event_columns = ['OS','TumorSpecificSurvival','MET','LR','RR','LRR','DFS'] # DFS here is not correct
event_columns_code = ['OS_code','TumorSpecificSurvival_code','MET_code','LR_code','RR_code','LRR_code','DFS_code'] 
survival_columns = ['TIME_OS','TIME_TumorSpecificSurvival','TIME_MET','TIME_LR','TIME_RR','TIME_LRR','TIME_DFS']

event_2year_columns = ['OS_2year','TumorSpecificSurvival_2year','MET_code_2year','LR_code_2year','RR_code_2year',
                        'LRR_code_2year','DFS_code_2year']
uncensoring_for2year_columns = ['OS_2year_uncensoring','TumorSpecificSurvival_2year_uncensoring','MET_code_2year_uncensoring',
                                'LR_code_2year_uncensoring','RR_code_2year_uncensoring','LRR_code_2year_uncensoring',
                                'DFS_code_2year_uncensoring'] # 1--uncensored, 0 --censored

#--------------------------------------------------
# investigation on levels of categorical variables
#--------------------------------------------------
# 1- draw Kaplan-Meier curves for different levels, and merge the ones having similar survival functions
'''
from lifelines import KaplanMeierFitter


for i in range(6):
    NoEp = i
    time_e = survival_columns[NoEp]
    event = event_columns_code[NoEp]

    # variable = 'Smoking_codes' 
    # levels = ['0-Current 191', '1-Never 172', '2-Past 243']
    # time point split-> levels = ['0-Current 177', '1-Never 123', '2-Past 39']
    # variable = 'MODALITY_codes' 
    # levels = ['0- RT 309', '1-Concurrent chemoradiation 297']
    # variable = 'TSTAD_codes' 
    # levels = ['0-T1 103','1-T2 198','2-T3 183','3-T4 122']
    # time poiint split -> levels = ['0-T1 54','1-T2 92','2-T3 53','3-T4 141']
    # variable = 'NSTAD_codes' 
    # levels = ['0-N0 101','1-N1 61','2-N2a 41','3-N2b 200','4-N2c 156','5-N3 47']
    # variable = 'P16_codes'
    # levels = ['0-unknown 24','1-negative 127','2-positive 98']

    # time point split -> levels = ['0-unknown 40','1-negative 176','2-positive 124']
    # variable = 'LOCTUM_codes'
    # traindata[variable].value_counts()
    # levels = ['0- Achter oroph 22', '1- Palatum 20', '2- Tongbasis 73','3- combi4locs 126','4- Vallecula 8']
    # time point ->levels = ['0- Achter oroph 34', '1- Palatum 34', '2- Tongbasis 88','3- combi4locs 173','4- Vallecula 11']
    variable = 'WHO_SCORE_codes'
    levels = ['0- ecog0 395', '1-ecog1 150', '2-ecog2 47', '3-ecog3 9', '4-ecog4 1','5-ecogNon 4']
    # time point ->levels = ['0- who0 227', '1-who1 91', '2-who2 19', '3-who3 3']
    

    ix_c0 = opcradiomics[variable] == 0
    ix_c1 = opcradiomics[variable] == 1
    ix_c2 = opcradiomics[variable] == 2
    ix_c3 = opcradiomics[variable] == 3
    ix_c4 = opcradiomics[variable] == 4
    ix_c5 = opcradiomics[variable] == 5
    
    plt.figure()
    ax = plt.subplot(111)
    kmf_0 = KaplanMeierFitter()
    ax = kmf_0.fit(opcradiomics.loc[ix_c0][time_e], opcradiomics.loc[ix_c0][event], label=levels[0]).plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    kmf_1 = KaplanMeierFitter()
    ax = kmf_1.fit(opcradiomics.loc[ix_c1][time_e], opcradiomics.loc[ix_c1][event], label=levels[1]).plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    kmf_2 = KaplanMeierFitter()
    ax = kmf_2.fit(opcradiomics.loc[ix_c2][time_e], opcradiomics.loc[ix_c2][event], label=levels[2]).plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    kmf_3 = KaplanMeierFitter()
    ax = kmf_3.fit(opcradiomics.loc[ix_c3][time_e], opcradiomics.loc[ix_c3][event], label=levels[3]).plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    kmf_4 = KaplanMeierFitter()
    ax = kmf_4.fit(opcradiomics.loc[ix_c4][time_e], opcradiomics.loc[ix_c4][event], label=levels[4]).plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    kmf_5 = KaplanMeierFitter()
    ax = kmf_5.fit(opcradiomics.loc[ix_c5][time_e], opcradiomics.loc[ix_c5][event], label=levels[5]).plot_survival_function(ax=ax,show_censors=True,ci_show=False)

    # add_at_risk_counts(kmf_exp, kmf_control, ax=ax)
    plt.tight_layout()
    plt.ylim([0.0,1.0])
    plt.title('KM curve --'+event)
    # plt.show()
    figname = "Z:\\Jiapan\\Radiotherapy\\OPC-Radiomics\\variablmerge\\" + variable +"_"+ event +'.png'
    plt.savefig(figname)

'''

all_colums_tosave = ['Trial PatientID']
all_colums_tosave.extend(clinical_para)
# all_colums_tosave.extend(clinical_para_binary)
all_colums_tosave.extend(event_columns_code)
all_colums_tosave.extend(survival_columns)
all_colums_tosave.extend(event_2year_columns)
all_colums_tosave.extend(uncensoring_for2year_columns)

OPCdigits = opcradiomics[all_colums_tosave]

OPCdigits.to_csv ('./opcradiomics_digits_202103.csv', index = True, header=True)
