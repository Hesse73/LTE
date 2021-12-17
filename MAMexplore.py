#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:49:40 2021

@author: caoyukun
"""

from Submodule_models import InputLoading,QueryEmbedding,TupleEmbedding,CFMAM
from Memorys import FeatureMem,QueryMem
from Main_models import user_mem_init, LOCALUpdate,BASEModel,maml_train,LOCALUpdate_offline,user_mem_init_offline

from utils import grads_sum,ConvexSpace,PolygonSpace
import torch
import warnings
import random
import  pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
     
def plotMatrixPoint(Mat, Label):
    print(Label)
    """
    :param Mat: 二维点坐标矩阵
    :param Label: 点的类别标签
    :return:
    """
    x = Mat[:, 0]
    y = Mat[:, 1]
    map_size = {1: 5, 0: 5}
    size = list(map(lambda x: map_size[x], Label))
    map_color = {1: 'r', 0: 'g'}
    color = list(map(lambda x: map_color[x], Label))
    map_marker = {1: 'o', 0: 'o'}
    markers = list(map(lambda x: map_marker[x], Label))
    # 下面一行代码会出错，因为marker参数不支持列表
    #  plt.scatter(np.array(x), np.array(y), s=size, c=color, marker=markers)
    # 下面一行代码为修正过的代码
    
    #plt.axis([0, 80000, 0, 500])

    plt.scatter(np.array(x), np.array(y), s=size, c=color, marker='o')  # scatter函数只支持array类型数据
    
    plt.show()   
warnings.filterwarnings('ignore')

class MAMexplore:
    def __init__(self,support_size,query_size,n_epoch,n_inner_loop,
                 batch_size,n_layer,embedding_dim,rho,lamda,
                 tao,cuda_option,n_k,alpha,beta,gamma,active_func,
                 train_task_num,test_task_num,path,query_vector_dim,
                 tuple_vector_dim,query_loading_dim,tuple_loading_dim):
        
        self.support_size = support_size
        self.query_size = query_size
        self.n_epoch = n_epoch
        self.n_inner_loop = n_inner_loop
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.embedding_dim = embedding_dim
        self.rho = rho  # local learning rate
        self.lamda = lamda # global learning rate
        self.tao = tao  # hyper-parameter for initializing personalized u weights
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(cuda_option if self.USE_CUDA else "cpu")
        self.n_k = n_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.active_func = active_func
        self.train_task_num=train_task_num
        self.test_task_num=test_task_num
        self.path=path
        self.query_vector_dim=query_vector_dim
        self.tuple_vector_dim=tuple_vector_dim
        self.query_loading_dim=query_loading_dim
        self.tuple_loading_dim=tuple_loading_dim
        
        # load dataset
        self.train_task_ids = list(range(self.train_task_num))
        
        self.test_task_ids = list(range(self.train_task_num,self.train_task_num+self.test_task_num))

        self.n_y = 2
    
        self.query_vector_loading, self.tuple_vector_loading = InputLoading(self.query_vector_dim,self.query_loading_dim).to(self.device), \
                                               InputLoading(self.tuple_vector_dim,self.tuple_loading_dim).to(self.device)

        # Embedding model
        self.QEmb = QueryEmbedding(self.n_layer, self.query_loading_dim ,
                                  self.embedding_dim, activation=self.active_func).to(self.device)
        self.TEmb = TupleEmbedding(self.n_layer, self.tuple_loading_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)

        # Classification_model 
        self.cf_model = CFMAM(self.embedding_dim, self.n_y, self.n_layer, activation=self.active_func).to(self.device)

        # whole model
        self.model = BASEModel(self.query_vector_loading, self.tuple_vector_loading,  self.QEmb, self.TEmb, self.cf_model).to(self.device)
        
        #分别为query和tuple embedding层的参数，以及分类模型mem层的参数
        
        self.phi_q, self.phi_t, self.phi_c = self.model.get_weights()

        self.FeatureMEM = FeatureMem(self.n_k, self.query_loading_dim,
                                     self.model, device=self.device)
        self.QueryMEM = QueryMem(self.n_k, self.embedding_dim, device=self.device)

        self.train = self.train_with_meta_optimization
        self.test = self.test_with_meta_optimization
        self.train()

    def train_with_meta_optimization(self):
        for i in range(self.n_epoch):
            q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()

            # On training dataset
            for index in self.train_task_ids:
                # init local parameters: theta_q, theta_t, theta_c
                '''这里还没改'''
                bias_term, att_values = user_mem_init(index, self.path, self.device, self.FeatureMEM, self.query_vector_loading,
                                                      self.alpha)
                self.model.init_q_mem_weights(self.phi_q, bias_term, self.tao, self.phi_t, self.phi_c)
                self.model.init_qt_mem_weights(att_values, self.QueryMEM)

                query_module = LOCALUpdate(self.model, index, self.support_size, self.query_size, self.batch_size,
                                          self.n_inner_loop, self.rho,self.path,device=self.device)
                q_grad, t_grad, c_grad = query_module.train()
                del query_module
                q_grad_sum, t_grad_sum, c_grad_sum = grads_sum(q_grad_sum, q_grad), grads_sum(t_grad_sum, t_grad), \
                                                     grads_sum(c_grad_sum, c_grad)
                with torch.no_grad(): 
                    self.FeatureMEM.write_head(q_grad, self.beta)
                    q_mqt = self.model.get_qt_mem_weights()
                    self.QueryMEM.write_head(q_mqt[0], self.gamma)
            
            self.phi_q, self.phi_t, self.phi_c = maml_train(self.phi_q, self.phi_t, self.phi_c,
                                                            q_grad_sum, t_grad_sum, c_grad_sum, self.lamda)
            del q_grad_sum, t_grad_sum, c_grad_sum
            print("!!!!!!!!!")
            self.test_with_meta_optimization()
#    def train_with_meta_optimization(self):
#        for i in range(self.n_epoch):
#            #q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()
#
#            # On training dataset
#            q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()
#            for index in self.train_task_ids:
#                #q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()
#                # init local parameters: theta_q, theta_t, theta_c
#                '''这里还没改'''
#                bias_term, att_values = user_mem_init(index, self.path, self.device, self.FeatureMEM, self.query_vector_loading,
#                                                      self.alpha)
#                self.model.init_q_mem_weights(self.phi_q, bias_term, self.tao, self.phi_t, self.phi_c)
#                self.model.init_qt_mem_weights(att_values, self.QueryMEM)
#
#                query_module = LOCALUpdate(self.model, index, self.support_size, self.query_size, self.batch_size,
#                                          self.n_inner_loop, self.rho,device=self.device)
#                q_grad, t_grad, c_grad = query_module.train()
#                del query_module
#                q_grad_sum, t_grad_sum, c_grad_sum = grads_sum(q_grad_sum, q_grad), grads_sum(t_grad_sum, t_grad), \
#                                                     grads_sum(c_grad_sum, c_grad)
#                self.FeatureMEM.write_head(q_grad, self.beta)
#                q_mqt = self.model.get_qt_mem_weights()
#                self.QueryMEM.write_head(q_mqt[0], self.gamma)
#             
#
#               
#                if index%20==0:    
##                    self.FeatureMEM.write_head(q_grad, self.beta)
##                    q_mqt = self.model.get_qt_mem_weights()
##                    self.QueryMEM.write_head(q_mqt[0], self.gamma)
#                    self.phi_q, self.phi_t, self.phi_c = maml_train(self.phi_q, self.phi_t, self.phi_c,
#                                                                    q_grad_sum, t_grad_sum, c_grad_sum, self.lamda)
#                    q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()
#            #del q_grad_sum, t_grad_sum, c_grad_sum
#            print("!!!!!!!!!")
#            self.test_with_meta_optimization()
#    def train_with_meta_optimization(self):
#        for i in range(self.n_epoch):
#            #q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()
#            batch_train_task_ids=random.sample(self.train_task_ids,32)
#            # On training dataset
#            q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()
#            for index in batch_train_task_ids:
#                q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()
#                # init local parameters: theta_q, theta_t, theta_c
#                '''这里还没改'''
#                bias_term, att_values = user_mem_init(index, self.path, self.device, self.FeatureMEM, self.query_vector_loading,
#                                                      self.alpha)
#                self.model.init_q_mem_weights(self.phi_q, bias_term, self.tao, self.phi_t, self.phi_c)
#                self.model.init_qt_mem_weights(att_values, self.QueryMEM)
#
#                query_module = LOCALUpdate(self.model, index, self.support_size, self.query_size, self.batch_size,
#                                          self.n_inner_loop, self.rho,device=self.device)
#                q_grad, t_grad, c_grad = query_module.train()
#                del query_module
#                q_grad_sum, t_grad_sum, c_grad_sum = grads_sum(q_grad_sum, q_grad), grads_sum(t_grad_sum, t_grad), \
#                                                     grads_sum(c_grad_sum, c_grad)
#                self.FeatureMEM.write_head(q_grad, self.beta)
#                q_mqt = self.model.get_qt_mem_weights()
#                self.QueryMEM.write_head(q_mqt[0], self.gamma)
#             
#
#               
#                #if index%10==0:                    
#            self.phi_q, self.phi_t, self.phi_c = maml_train(self.phi_q, self.phi_t, self.phi_c,
#                                                                    q_grad_sum, t_grad_sum, c_grad_sum, self.lamda)
#                    #q_grad_sum, t_grad_sum, c_grad_sum = self.model.get_zero_weights()
#            #del q_grad_sum, t_grad_sum, c_grad_sum
#            print("!!!!!!!!!")
#            self.test_with_meta_optimization()

    def test_with_meta_optimization(self):
        best_phi_q, best_phi_t, best_phi_c = self.model.get_weights()

        for index in self.test_task_ids:
            bias_term, att_values = user_mem_init(index, self.path, self.device, self.FeatureMEM, self.query_vector_loading,
                                                      self.alpha)
            self.model.init_q_mem_weights(best_phi_q, bias_term, self.tao, best_phi_t, best_phi_c)
            self.model.init_qt_mem_weights(att_values, self.QueryMEM)

            self.model.init_weights(best_phi_q, best_phi_t, best_phi_c)
            query_module = LOCALUpdate(self.model, index,  self.support_size, self.query_size, self.batch_size,
                                      self.n_inner_loop, self.rho,self.path,device=self.device)
            
            query_module.test()
            del query_module
            
    def test_with_offline_task(self,offline_task_ids,group_id,path,support_size,query_size,batch_size,n_inner_loop,attrg_id,dim):
        #task_map=pickle.load(open('{}/'.format(path2)+'task_map.p', 'rb'))
#        DSmodel=torch.load("DS_test_2D.bin")
#        DSmodel.DataCenters_task_index=DSmodel.DataCenters_task_space
#        DSmodel.queryspace_taskindex_centers_neighbors=DSmodel.queryspace_centers_neighbors

        solve={}
        for index in offline_task_ids:
            solve[index]={}
        
        best_phi_q, best_phi_t, best_phi_c = self.model.get_weights()
        for index in offline_task_ids:
            print('!!!!!!!!:',index)
            #print(best_phi_q)
            bias_term, att_values = user_mem_init_offline(index, group_id,path, self.device, self.FeatureMEM, self.query_vector_loading,
                                                          self.alpha,dim)
            self.model.init_q_mem_weights(best_phi_q, bias_term, self.tao, best_phi_t, best_phi_c)
            self.model.init_qt_mem_weights(att_values, self.QueryMEM)
    
            self.model.init_weights(best_phi_q, best_phi_t, best_phi_c)
            query_module = LOCALUpdate_offline(self.model, group_id,index, support_size,query_size,batch_size,
                                          n_inner_loop,self.rho,path,self.device,dim)
            
            ture,pred,Accuracy,Recall,Precision,F1_score=query_module.test()
            
            solve[index]['pred']=pred
            solve[index]['Accuracy']=Accuracy
            solve[index]['Recall']=Recall
            solve[index]['Precision']=Precision
            solve[index]['F1_score']=F1_score
        return solve

    def test_with_offline_task_opt(self,offline_task_ids,group_id,path,support_size,query_size,batch_size,n_inner_loop,attrg_id,dim):
        num=path.split("_")[-1]
        print(path)
        print(num)
        print(group_id)
        print(attrg_id)
        if 'mode' in num:
            DSmodel=torch.load("DS2_model_nomal_service.bin")
        else:
            DSmodel=torch.load("DS2_model_nomal_service_"+str(num)+".bin")
        #task_map=pickle.load(open('{}/'.format(path2)+'task_map.p', 'rb'))
#        DSmodel=torch.load("DS_test_2D.bin")
        DSmodel.DataCenters_task_index=DSmodel.DataCenters_task_space
        DSmodel.queryspace_taskindex_centers_neighbors=DSmodel.queryspace_centers_neighbors
        

        solve={}
        for index in offline_task_ids:
            solve[index]={}
        
        best_phi_q, best_phi_t, best_phi_c = self.model.get_weights()
        for index in offline_task_ids:
            print('!!!!!!!!:',index)
            #print(best_phi_q)
            bias_term, att_values = user_mem_init_offline(index, group_id,path, self.device, self.FeatureMEM, self.query_vector_loading,
                                                          self.alpha,dim)
            self.model.init_q_mem_weights(best_phi_q, bias_term, self.tao, best_phi_t, best_phi_c)
            self.model.init_qt_mem_weights(att_values, self.QueryMEM)
    
            self.model.init_weights(best_phi_q, best_phi_t, best_phi_c)
            query_module = LOCALUpdate_offline(self.model, group_id,index, support_size,query_size,batch_size,
                                          n_inner_loop,self.rho,path,self.device,dim)
            
        
            ture,pred,Accuracy,Recall,Precision,F1_score=query_module.test()
            
            if 'mode' in num:
                if len(query_module.center_index)/25<0.5:
                    step1=25
                    step2=10
                else:
                    step1=30
                    step2=15

                    
            elif int(num)==50:
                if len(query_module.center_index)/25<0.5:
                    step1=25
                    step2=8
                else:
                    step1=30
                    step2=10

            elif int(num)==75:
                if len(query_module.center_index)/25<0.5:
                    step1=20
                    step2=5
                else:
                    step1=25
                    step2=10
     
            elif int(num)==100:
                if len(query_module.center_index)/25<0.5:
                    step1=20
                    step2=5
                else:
                    step1=25
                    step2=8
      
             
            
            cspace=[]
           
            #print(query_module.center_index)
            #plotMatrixPoint(DSmodel.numpy_sample_raw[:,DSmodel.pos_list_offline[attrg_id]],np.array(ture)) 
            #plotMatrixPoint(DSmodel.numpy_sample_raw[:,DSmodel.pos_list_offline[attrg_id]],np.array(pred)) 
            for i in query_module.center_index:
                
                temp_samples=DSmodel.DataCenters_task_index[attrg_id][DSmodel.queryspace_taskindex_centers_neighbors[attrg_id][i][:step1]]
                #print(DSmodel.queryspace_taskindex_centers_neighbors[i][:10])
                #print([DSmodel.DataCenters_queryspace[i]])
                temp_samples=np.append(temp_samples,[DSmodel.DataCenters_queryspace[attrg_id][i]],0)
                #plotMatrixPoint(temp_samples,np.zeros(len(temp_samples),))
                cspace.append(ConvexSpace(temp_samples))
            #cspace.append(ConvexSpace(DSmodel.DataCenters_queryspace[attrg_id][query_module.center_index]))
            
            
            cppace=[]
            
            
            for i in query_module.center_index:
                #print(i)
                temp_samples=DSmodel.DataCenters_task_index[attrg_id][DSmodel.queryspace_taskindex_centers_neighbors[attrg_id][i][:step2]]
                #print(DSmodel.queryspace_taskindex_centers_neighbors[i][:10])
                #print([DSmodel.DataCenters_queryspace[i]])
                temp_samples=np.append(temp_samples,[DSmodel.DataCenters_queryspace[attrg_id][i]],0)
                #plotMatrixPoint(temp_samples,np.zeros(len(temp_samples),))
                cppace.append(PolygonSpace(temp_samples))
            #cppace.append(PolygonSpace(DSmodel.DataCenters_queryspace[attrg_id][query_module.center_index]))

            
            
            
            
            
            f1=[]
            pred=pred.numpy()
            
            pos_indexs=np.where(pred==1)[0]
            neg_indexs=np.where(pred==0)[0]
            
            
            pos_labels=[]
            neg_labels=[]
            for i in range(len(pos_indexs)):
                result=False
                for j in cspace:
                    if j.in_pos_region(DSmodel.numpy_sample_raw[pos_indexs[i]][DSmodel.pos_list_offline[attrg_id]]):
                        result=True
                        break
                pos_labels.append(result)
            
            pred_1=pred.copy()
            pred_1[pos_indexs]=pos_labels
            
            print(f1_score(ture,pred_1))
            f1.append(f1_score(ture,pred_1))
            #plotMatrixPoint(DSmodel.numpy_sample_raw[:,DSmodel.pos_list_offline[attrg_id]],pred_1) 
            
            if attrg_id!=3:
            
                for i in range(len(neg_indexs)):
                    result=False
                    for j in cppace:
                        if j.in_pos_region(DSmodel.numpy_sample_raw[neg_indexs[i]][DSmodel.pos_list_offline[attrg_id]]):
                            result=True
                            break
                    neg_labels.append(result)
                
                pred_2=pred.copy()
                pred_2[neg_indexs]=neg_labels
                print(f1_score(ture,pred_2))
                f1.append(f1_score(ture,pred_2))
                #plotMatrixPoint(DSmodel.numpy_sample_raw[:,DSmodel.pos_list_offline[attrg_id]],pred_2)    
                
                
                pos_indexs2=np.where(pred_2==1)[0]
                pos_labels2=[]
                
                for i in range(len(pos_indexs2)):
                    result=False
                    for j in cspace:
                        if j.in_pos_region(DSmodel.numpy_sample_raw[pos_indexs2[i]][DSmodel.pos_list_offline[attrg_id]]):
                            result=True
                            break
                    pos_labels2.append(result)
                
                pred_3=pred_2.copy()
                pred_3[pos_indexs2]=pos_labels2
                print(f1_score(ture,pred_3))
                f1.append(f1_score(ture,pred_3))
                #plotMatrixPoint(DSmodel.numpy_sample_raw[:,DSmodel.pos_list_offline[attrg_id]],pred_3)    
                
                solve[index]['pred']=torch.tensor(pred_3)
                solve[index]['pred1']=torch.tensor(pred_1)
                solve[index]['pred2']=torch.tensor(pred_2)
                
                solve[index]['Accuracy']=Accuracy
                solve[index]['Recall']=Recall
                solve[index]['Precision']=Precision
                solve[index]['F1_score']=f1[2]
                solve[index]['F1_score_1']=f1[0]
                solve[index]['F1_score_2']=f1[1]
            else:
                solve[index]['pred']=torch.tensor(pred_1)
                solve[index]['pred1']=None
                solve[index]['pred2']=None
                
                solve[index]['Accuracy']=Accuracy
                solve[index]['Recall']=Recall
                solve[index]['Precision']=Precision
                solve[index]['F1_score']=f1[0]
                solve[index]['F1_score_1']=None
                solve[index]['F1_score_2']=None

             
            
                 
            
#            
#            
#            test_labels=[]
#            
#            
#            
#            
#            
#            f1=[]
#            for i in range(len(DSmodel.numpy_sample_raw)):
#                result=False
#                for j in cspace:
#                    if j.in_pos_region(DSmodel.numpy_sample_raw[i][DSmodel.pos_list_offline[attrg_id]]):
#                        result=True
#                        break
#                test_labels.append(result)
##            plotMatrixPoint(DSmodel.numpy_sample_raw[:,DSmodel.pos_list_offline[attrg_id]],np.array(test_labels))     
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,np.array(test_labels))    
#            print("covex:")
#            print(f1_score(ture,np.array(test_labels)))
#            f1.append(f1_score(ture,np.array(test_labels)))
#             
#            test_labels2=[]
#            for i in range(len(DSmodel.numpy_sample_raw)):
#                result=False
#                for j in cppace:
#                    if j.in_pos_region(DSmodel.numpy_sample_raw[i][DSmodel.pos_list_offline[attrg_id]]):
#                        result=True
#                        break
#                test_labels2.append(result)
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,np.array(test_labels2))    
##            plotMatrixPoint(DSmodel.numpy_sample_raw[:,DSmodel.pos_list_offline[attrg_id]],np.array(test_labels2))     
#            f1.append(f1_score(ture,np.array(test_labels2)))
#            
#            print("pylgon:")
#            print(f1_score(ture,np.array(test_labels2)))
#            
#            pred2 =pred & np.array(test_labels)
#            
#            print("covex+:")
#            print(f1_score(ture,pred2))
#            
#            f1.append(f1_score(ture,pred2))
#            
#           
#            pred3 =pred | np.array(test_labels2)
#            
#            print("pylgon+:")
#            print(f1_score(ture,pred3))
#            
#            f1.append(f1_score(ture,pred3))
#            
#            
#            
#            pred4 = pred2| np.array(test_labels2)
#            print("final:")
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,pred)
#            print(f1_score(ture,pred4))
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,pred)
#            f1.append(f1_score(ture,pred4))
#            #all_f12.append(f1_score(ture,pred))

            
       
            
            
        return solve

            #print(np.where(querysapce_vec==1))
        
#            plotMatrixPoint(DSmodel.numpy_sample_raw,ture.numpy())
#            plotMatrixPoint(DSmodel.numpy_sample_raw,pred.numpy())
#            cspace=[]
#            #plotMatrixPoint(DSmodel.DataCenters_queryspace[center_index],np.zeros(len(center_index),))
#            
#            for i in center_index:
#                #print(i)
#                temp_samples=DSmodel.DataCenters_task_index[DSmodel.queryspace_taskindex_centers_neighbors[i][:15]]
#                #print(DSmodel.queryspace_taskindex_centers_neighbors[i][:10])
#                #print([DSmodel.DataCenters_queryspace[i]])
#                temp_samples=np.append(temp_samples,[DSmodel.DataCenters_queryspace[i]],0)
#                #plotMatrixPoint(temp_samples,np.zeros(len(temp_samples),))
#                cspace.append(ConvexSpace(temp_samples))
#            cspace.append(ConvexSpace(DSmodel.DataCenters_queryspace[center_index]))
#            
#            
#            cppace=[]
#            
#            
#            for i in center_index:
#                #print(i)
#                temp_samples=DSmodel.DataCenters_task_index[DSmodel.queryspace_taskindex_centers_neighbors[i][:5]]
#                #print(DSmodel.queryspace_taskindex_centers_neighbors[i][:10])
#                #print([DSmodel.DataCenters_queryspace[i]])
#                temp_samples=np.append(temp_samples,[DSmodel.DataCenters_queryspace[i]],0)
#                #plotMatrixPoint(temp_samples,np.zeros(len(temp_samples),))
#                cppace.append(PolygonSpace(temp_samples))
#            #cspace.append(ConvexSpace(DSmodel.DataCenters_queryspace[center_index]))
#
#            
#            test_labels=[]
#            
#            
#            
#            
#            for i in range(len(DSmodel.numpy_sample_raw)):
#                result=False
#                for j in cspace:
#                    if j.in_pos_region(DSmodel.numpy_sample_raw[i]):
#                        result=True
#                        break
#                test_labels.append(result)
#            plotMatrixPoint(DSmodel.numpy_sample_raw,np.array(test_labels))    
#            #print(f1_score(ture,np.array(test_labels)))
#             
#            test_labels2=[]
#            for i in range(len(DSmodel.numpy_sample_raw)):
#                result=False
#                for j in cppace:
#                    if j.in_pos_region(DSmodel.numpy_sample_raw[i]):
#                        result=True
#                        break
#                test_labels2.append(result)
#            plotMatrixPoint(DSmodel.numpy_sample_raw,np.array(test_labels2))    
#            
#            
#            pred =pred.numpy() & np.array(test_labels)
#            print(f1_score(ture,pred))
#            pred =pred | np.array(test_labels2)
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,pred)
#            print(f1_score(ture,pred))
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,pred)
#            
#            all_f12.append(f1_score(ture,pred))
            
                
                
                
            
            
            
#            
#            #plotMatrixPoint(DSmodel.DataCenters_task_index,task_index_label.numpy())
#            
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,ture.numpy())
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,pred.numpy())
#               
#            task_index_label=querysapce_vec
#            
#            all_acc.append(Accuracy)
#            all_rec.append(Recall)
#            all_pre.append(Precision)
#            all_f1.append(F1_score)
#
#            cluster_samples=[]
#            index=0
#            for i in task_index_label:
#                if i ==1:
#                    cluster_samples.append(DSmodel.DataCenters_task_index[index])
#                   
#                index+=1
#            cluster_samples= DSmodel.model.transform(pd.DataFrame(np.array(cluster_samples),columns= DSmodel.raw.columns)) 
#            kmodel=KMeans(cluster_num)
#            kmodel.fit(cluster_samples)
#            cspace=[]
#            for i in range(len(kmodel.cluster_centers_)):
#                
#                temp_samples=cluster_samples[np.where(kmodel.labels_==i)]
#                temp_samples=DSmodel.model.inverse_transform(temp_samples,None).to_numpy()
#                #plotMatrixPoint(temp_samples,np.zeros(len(temp_samples),))
#                cspace.append(ConvexSpace(temp_samples))
#            
#            test_labels=[]
#            
#            
#            
#            
#            for i in range(len(DSmodel.numpy_sample_raw)):
#                result=False
#                for j in cspace:
#                    if j.in_pos_region(DSmodel.numpy_sample_raw[i]):
#                        result=True
#                        break
#                test_labels.append(result)
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,np.array(test_labels))    
#            #print(f1_score(ture,np.array(test_labels)))
#             
#            
#            
#            pred =pred.numpy() & np.array(test_labels)
#            #plotMatrixPoint(DSmodel.numpy_sample_raw,pred)
#            print(f1_score(ture,pred))
#            
#            
#            all_f12.append(f1_score(ture,pred))
            
            
            
#            print(sum(task_index_label))
#            task_index_label=task_index_label.numpy()
#           
#            sum1=sum(task_index_label)
#
#            focus_num=int(sum1*1)
#            temp=np.where(task_index_label==1)[0]
#            
#            choose_task_sets=[]
#            plotMatrixPoint(DSmodel.numpy_sample_raw,ture.numpy())
#            plotMatrixPoint(DSmodel.numpy_sample_raw,pred.numpy())
#            
#            for times in range(5):
#                
#                sample_index=np.random.choice(temp,focus_num,replace=False)
##                
##                FixSpace=PolygonSpace(DSmodel.DataCenters_task_index[sample_index])
##                
##                temp_index=0
##                fix_map=[]
##                FixSapce_labels=[]
##                for i in range(len(DSmodel.numpy_sample_raw)):
##                    FixSapce_labels.append(FixSpace.in_pos_region(DSmodel.numpy_sample_raw[i]))
##                    if FixSpace.in_pos_region(DSmodel.numpy_sample_raw[i]):
##                        fix_map.append(temp_index)
##                        
##                    temp_index+=1
##                    
##                print(np.array(FixSapce_labels))    
##                plotMatrixPoint(DSmodel.numpy_sample_raw,np.array(FixSapce_labels))
#                #print(fix_map)
#                task_index_label_sample=np.zeros(len(task_index_label),)
#                
#                task_index_label_sample[sample_index]=1
#                
#                
#                
#                '''
#                搜索匹配过程
#                '''
#                temp_tasks=set(range(100000))
#                index=0
#                for i in task_index_label_sample:
#                    if i==1:
#                        temp_tasks=set(task_map[index])& temp_tasks
#                    index+=1
#                   
#                temp_tasks=list(temp_tasks)
#                
#                                    
#                if len(temp_tasks)!=0:
#                    if len(temp_tasks)<=5:
#                        choose_task_sets=choose_task_sets+temp_tasks
#                    else:
#                        choose_task_sets=choose_task_sets+temp_tasks[:4]
#                
#                
#            pred=pred.numpy()
#            if len(choose_task_sets)!=0:
#                
#                for task_id in choose_task_sets:
#                    test_labels=[]
#                    tiv,qps=load_offline_task_info(task_id,path2)
#                    
#                    for i in range(len(DSmodel.numpy_sample_raw)):
#    
#                        result=False
#                        for j in qps:
#                            if j.in_pos_region(DSmodel.numpy_sample_raw[i]):
#                                result=True
#                                break
#                        test_labels.append(result)
#                    plotMatrixPoint(DSmodel.numpy_sample_raw,np.array(test_labels))    
#                    print(f1_score(ture,np.array(test_labels)))
#                        
#                    print(f1_score(ture,pred))
#                    pred =pred & np.array(test_labels)
#                    #pred[fix_map]=pred[fix_map] & np.array(test_labels)[fix_map] 
#                    plotMatrixPoint(DSmodel.numpy_sample_raw,pred)
#                print(f1_score(ture,pred))
#                
#                all_f12.append(f1_score(ture,pred))
#                        
#            else:
#                all_f12.append(F1_score)
#            
            
              
       

