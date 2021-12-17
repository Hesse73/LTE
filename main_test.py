#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:25:45 2021

@author: caoyukun
"""

from MAMexplore import MAMexplore
from DataSpace import DataSpace
from TaskGenerator_Offline import TaskGeneratorOffline
from GlobalConfigs2 import OfflineTaskGenerate_configs
import pandas as pd
import torch
import pickle
data=pd.read_csv('SDSS_10w.csv')  
data=data[['rowc', 'colc','sky_u','sky_g','ra','dec','nDetect','nEdge']].astype('float32') 

               
DSmodel=torch.load("DS2_model_nomal_service.bin")

mode_list={'1':{'attr_list':[['rowc', 'colc']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
            '2':{'attr_list':[['ra','dec']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
            '3':{'attr_list':[['nDetect','nEdge']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},   
                 
            '4':{'attr_list':[['rowc', 'colc'],['ra','dec']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
                 
            '5':{'attr_list':[['rowc', 'colc'],['nDetect','nEdge']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
            '6':{'attr_list':[['ra','dec'],['nDetect','nEdge']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
                 
            '7':{'attr_list':[['rowc', 'colc'],['ra','dec']],
                'u_flag':[0],
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
                 
            '8':{'attr_list':[['rowc', 'colc'],['nDetect','nEdge']],
                'u_flag':[0],
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
                 
            '9':{'attr_list':[['rowc', 'colc'],['ra','dec'],['nDetect','nEdge']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
            '10':{'attr_list':[['sky_u','sky_g'],['ra','dec'],['nDetect','nEdge']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},   
                  
            '11':{'attr_list':[['rowc', 'colc'],['ra','dec'],['nDetect','nEdge']],
                'u_flag':[0],
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
            '12':{'attr_list':[['sky_u','sky_g'],['ra','dec'],['nDetect','nEdge']],
                'u_flag':[1],
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},     
            '13':{'attr_list':[['rowc', 'colc'],['sky_u','sky_g'],['ra','dec'],['nDetect','nEdge']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
                  
            '14':{'attr_list':[['rowc', 'colc'],['sky_u','sky_g'],['ra','dec'],['nDetect','nEdge']],
                'u_flag':[1],
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
                  
            '15':{'attr_list':[['rowc', 'colc'],['sky_u','sky_g'],['ra','dec'],['nDetect','nEdge']],
                'u_flag':[2],
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},

            '16':{'attr_list':[['sky_u','sky_g']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
            '17':{'attr_list':[['rowc', 'colc'],['sky_u','sky_g']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
                 
            '18':{'attr_list':[['sky_u','sky_g'],['ra','dec']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
            '19':{'attr_list':[['rowc', 'colc'],['sky_u','sky_g'],['ra','dec']],
                'u_flag':None,
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]},
            '20':{'attr_list':[['rowc', 'colc'],['sky_u','sky_g'],['ra','dec']],
                'u_flag':[0],
                'reg':[[4,10],[4,20],[3,10],[3,20],[2,10],[2,20]]}
}


            
            
            
random_sample_num=30
for m in mode_list.keys():
    TGmodel=TaskGeneratorOffline(
            DSmodel,
            OfflineTaskGenerate_configs['task_num'],
            OfflineTaskGenerate_configs['support_tuple_num'],
            OfflineTaskGenerate_configs['intrval_flag'],
            OfflineTaskGenerate_configs['space_list'],
            OfflineTaskGenerate_configs['path']+"_mode"+str(m),
            OfflineTaskGenerate_configs['task_complexity'],
            OfflineTaskGenerate_configs['taskspace_topk'],
            OfflineTaskGenerate_configs['queryspace_topk'],
            OfflineTaskGenerate_configs['multi_intrval'],
            mode_list[m],
            random_sample_num)
        
    TGmodel.build(None)
    
label_num=[50,75,100]
for num in  label_num:
   DSmodel=torch.load("DS2_model_nomal_service_"+str(num)+".bin")
   for m in mode_list.keys():
       qps=pickle.load(open('{}/qps.p'.format(OfflineTaskGenerate_configs['path']+"_mode"+str(m)), 'rb'))
       TGmodel=TaskGeneratorOffline(
                DSmodel,
                OfflineTaskGenerate_configs['task_num'],
                num+5,
                OfflineTaskGenerate_configs['intrval_flag'],
                OfflineTaskGenerate_configs['space_list'],
                OfflineTaskGenerate_configs['path']+"_mode"+str(m)+'_'+str(num),
                OfflineTaskGenerate_configs['task_complexity'],
                OfflineTaskGenerate_configs['taskspace_topk'],
                OfflineTaskGenerate_configs['queryspace_topk'],
                OfflineTaskGenerate_configs['multi_intrval'],
                mode_list[m],
                30)
            
       TGmodel.build(qps)
   


