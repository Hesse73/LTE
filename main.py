#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:19:49 2021

@author: caoyukun
"""

from MAMexplore import MAMexplore
from DataSpace import DataSpace
from TaskGenerator import TaskGenerator
#from TaskGenerate3 import TaskGenerate_Implicit_method_multi

from GlobalConfigs import dataspace_configs,taskGenerate_configs,mamexplore_configs
import pandas as pd
import torch
            

data=pd.read_csv('SDSS_10w.csv')  
data=data[['rowc', 'colc','sky_u','sky_g','ra','dec','nDetect','nEdge']].astype('float32') 





DSmodel=DataSpace(
        data,
        dataspace_configs['cluster_method'],
        dataspace_configs['cluster_task_space_num'],
        dataspace_configs['cluster_queryspace_num'],
        dataspace_configs['cluster_queryset_space_num'],
        dataspace_configs['represent_method'],
        dataspace_configs['online_split_list'],
        dataspace_configs['offline_split_list'],
        dataspace_configs['cluster_sample_rate'],
        dataspace_configs['JK_GMM_sample_rate'])

         

DSmodel.build()    
torch.save(DSmodel,"DS2_model_nomal_service.bin")
 

TGmodel=TaskGenerator(
        DSmodel,
        taskGenerate_configs['train_task_num'],
        taskGenerate_configs['test_task_num'],
        taskGenerate_configs['support_tuple_num'],
        taskGenerate_configs['query_tuple_num'],
        taskGenerate_configs['split_list'],
        taskGenerate_configs['path'],
        taskGenerate_configs['task_complexity'],
        taskGenerate_configs['taskspace_topk'],
        taskGenerate_configs['queryspace_topk'])


TGmodel.build()  


torch.save(TGmodel,"TG_"+taskGenerate_configs['path']+".bin")

for i in range(len(TGmodel.split_list)):
    out_dim=0
    for j in TGmodel.split_list[i]:
        for n in TGmodel.dataspace.model.meta:
            if n['name']==j:
                out_dim+=n['output_dimensions']
                
                
    Main_model= MAMexplore(
        mamexplore_configs['support_size'],
        mamexplore_configs['query_size'],
        mamexplore_configs['n_epoch'],
        mamexplore_configs['n_inner_loop'],
        mamexplore_configs['batch_size'],
        mamexplore_configs['n_layer'],
        mamexplore_configs['embedding_dim'],
        mamexplore_configs['rho'],
        mamexplore_configs['lamda'],
        mamexplore_configs['tao'],
        mamexplore_configs['cuda_option'],
        mamexplore_configs['n_k'],
        mamexplore_configs['alpha'],
        mamexplore_configs['beta'],
        mamexplore_configs['gamma'],
        mamexplore_configs['active_func'],
        mamexplore_configs['train_task_num'],
        mamexplore_configs['test_task_num'],
        mamexplore_configs['path']+"/attr_group_"+str(i)+"/",
        mamexplore_configs['query_vector_dim'],
        out_dim,
        mamexplore_configs['query_loading_dim'],
        mamexplore_configs['tuple_loading_dim'])

    torch.save(Main_model,"Model_"+mamexplore_configs['path']+"_"+str(i))