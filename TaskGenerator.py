
import pandas as pd
import torchvision.transforms as transforms
import math
import shapely
from shapely.geometry import Point, LineString, MultiPoint
from numpy import float32,int64,float16

import pickle
import os.path
import numpy as np
import random
from math import sqrt, acos, pi
from numpy import cross
from utils import *
from DataSpace import *
import matplotlib.pylab as plt
'''这里实现3，4维的task生成情况
   我们预设属性都是固定分割好的，如果之后有需要可以再重构一个属性分割的类
   2维情况下：price和powerPS这两个属性绑定
   3维情况下：price和powerPS这两个属性绑定，Age单独作为一个属性
   4维情况下：price和powerPS这两个属性绑定，Age和kilometer绑定
   生成task的方案：不划分太细致的pattern类别，而是直接在低维度的空间上抽样点，并且去构造这些点组成的凸的region。
'''



def plotMatrixPoint(Mat, Label):
    """
    :param Mat: 二维点坐标矩阵
    :param Label: 点的类别标签
    :return:
    """
    x = Mat[:, 0]
    y = Mat[:, 1]
    map_size = {1: 10, 0: 10}
    size = list(map(lambda x: map_size[x], Label))
    map_color = {1: 'r', 0: 'g'}
    color = list(map(lambda x: map_color[x], Label))
    map_marker = {1: 'o', 0: 'o'}
    markers = list(map(lambda x: map_marker[x], Label))
    # 下面一行代码会出错，因为marker参数不支持列表
    #  plt.scatter(np.array(x), np.array(y), s=size, c=color, marker=markers)
    # 下面一行代码为修正过的代码
    
    plt.axis([0, 80000, 0, 500])

    plt.scatter(np.array(x), np.array(y), s=size, c=color, marker='o')  # scatter函数只支持array类型数据
    
    plt.show()
 

        
        
class TaskGenerator:

    def __init__(self, dataspace,train_task_num,test_task_num,support_tuple_num,query_tuple_num,split_list,path,task_complexity,taskspace_topk,queryspace_topk):

        self.dataspace= dataspace #数据空间对象
        self.train_task_num = train_task_num #要构造的训练task数目
        self.test_task_num =test_task_num #要构造的测试task数目
        
        self.n_way = 2
        self.k_shot = support_tuple_num#支撑集的样本个数，这里包括两部分聚类中心与随机抽样元组
        self.k_query= query_tuple_num #查询集的样本个数，这里与test数据空间的聚类数相同
        self.path=path #生成的task的数据文件的存储地址
        self.split_list=split_list 
        self.attr_dim=self.dataspace.raw.shape[1]
        self.task_sample_num=None #在子任务上构建凸空间或者多边形空间抽样的样本个数，要小于支撑集的样本个数
        self.task_complexity=task_complexity 
        '''
        task_complexity指构建任务的复杂性：
        n：指子空间上凸查询区域的最大值
        '''
        self.taskspace_topk=taskspace_topk
        self.queryspace_topk=queryspace_topk
        self.range=[[-2.972521,1445.488037,5.036839,2039.900024],[0.561423,272.201996,0.768829,170.067200],[0.009221,359.998444,-24.207066,84.823318],[1.000000,40.000000,0.000000,27.000000]]


    
    def task_2D(self,attr_id):
       
     
        region_num=self.task_complexity
            
        CovSpaces=[]
        indexlist=np.array(range(self.dataspace.cluster_task_space_num))
        for i in range(region_num):
            center=random.sample(list(indexlist),1)

            neighbors=self.dataspace.taskspace_centers_neighbors[attr_id][center[0]][:self.taskspace_topk+1]
            
            condition_data=self.dataspace.DataCenters_task_space[attr_id][neighbors]
#      
#            plt.axis(self.range[attr_id])
#            plt.scatter(condition_data[0][0],condition_data[0][1],color='yellow')
#            plt.scatter(np.array(condition_data)[:,0],np.array(condition_data)[:,1],s=15,color='black',marker='o')
#            plt.show()
            CovSpace=ConvexSpace(condition_data)
            CovSpaces.append(CovSpace)
            indexlist=np.setdiff1d(indexlist, neighbors)
            
#        plotdata=[]    
#        
#        for i in range(int(self.dataspace.numpy_sample_raw.shape[0]/100)):
#            for j in CovSpaces:
#                if j.in_pos_region(self.dataspace.numpy_sample_raw[:,self.dataspace.pos_list[attr_id]][i]):
#                    plotdata.append(self.dataspace.numpy_sample_raw[:,self.dataspace.pos_list[attr_id]][i])
#                    break
#        plt.axis(self.range[attr_id])
#        plt.scatter(np.array(plotdata)[:,0],np.array(plotdata)[:,1],s=15,color='black',marker='o')
#        plt.scatter(self.dataspace.DataCenters_queryspace[attr_id][:,0],self.dataspace.DataCenters_queryspace[attr_id][:,1],s=15,color='red',marker='o')
#        plt.scatter(self.dataspace.DataCenters_task_space[attr_id][:,0],self.dataspace.DataCenters_task_space[attr_id][:,1],s=15,color='green',marker='o')
#        plt.show()
        
        return CovSpaces
            
            
            
    def Generate_single(self,attr_gid):
        
            query_vector_ori=[]
            query_vector=np.zeros(self.dataspace.cluster_task_space_num)
            train_labels=[] 
            test_labels=[]
            train_tuples=[]
            test_tuples=[]     
     

            query_patterns=self.task_2D(attr_gid)
            hot_index=0
            hot_list=[]
            for i in self.dataspace.DataCenters_queryspace[attr_gid]:
                result=False
                for j in query_patterns:
                    if j.in_pos_region(i):
                        result=True
                        break
                query_vector_ori.append(result)
                train_tuples.append(i)
                train_labels.append(result)
                if result==True:
                    hot_list.append(hot_index)    
                hot_index+=1
                
            if len(hot_list)==0:
                return None,None,None,None,None,None,None
            
            
            print(hot_list)
            for i in hot_list:
                query_vector[self.dataspace.queryspace_centers_neighbors[attr_gid][i][:self.queryspace_topk]]=True
                
            print(query_vector)    
            
#            plt.axis([0, 80000, 0, 500])
#            plt.scatter(np.array(train_tuples)[:,0],np.array(train_tuples)[:,1])
#            plt.show()
#            
#            
            
            '''
            随机从训练与测试的聚类中抽样样本，进行标记,每个聚簇抽样一个
            '''    
            
            for i in range(self.k_shot-self.dataspace.cluster_queryspace_num):
                index=random.choice(hot_list)
                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryspace[attr_gid]==index)][:,self.dataspace.pos_list[attr_gid]]
                #print(temp_data)
                temp_tuple=random.sample(list(temp_data),1)
                
                train_tuples.append(temp_tuple[0]) 
                result=False
                for j in query_patterns:
                    if j.in_pos_region(temp_tuple[0]):
                        result=True
                        break
                train_labels.append(result)
                
            for i in range(self.k_query):
                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryset_space[attr_gid]==i)][:,self.dataspace.pos_list[attr_gid]]
                temp_tuple=random.sample(list(temp_data),1)
                
                test_tuples.append(temp_tuple[0]) 
                result=False
                for j in query_patterns:
                    if j.in_pos_region(temp_tuple[0]):
                        result=True
                        break

                test_labels.append(result)
            
            for i in range(int(self.k_query/2)):
                index=random.choice(hot_list)
                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryspace[attr_gid]==index)][:,self.dataspace.pos_list[attr_gid]]
                temp_tuple=random.sample(list(temp_data),1) 
                test_tuples.append(temp_tuple[0]) 
                result=False
                for j in query_patterns:
                    if j.in_pos_region(temp_tuple[0]):
                        result=True
                        break
                test_labels.append(result)


            train_tuples=np.array(train_tuples)
            
            
            test_tuples=np.array(test_tuples)
            #plt.scatter(train_tuples[:,0],train_tuples[:,1])
            #plt.scatter(test_tuples[:,0],test_tuples[:,1])
            train_labels=np.array(train_labels)
            test_labels=np.array(test_labels)
            
            #plotMatrixPoint(train_tuples, train_labels)
            #plotMatrixPoint(test_tuples, test_labels)
 
            query_vector=np.array(query_vector)
            train_rate=np.sum(train_labels==True)/len(train_labels)
            test_rate=np.sum(test_labels==True)/len(test_labels)
            all_attrs=self.dataspace.raw.columns.to_list()
            
            Temp_train_tuples=np.zeros([len(train_tuples),len(all_attrs)])
            
            Temp_test_tuples=np.zeros([len(test_tuples),len(all_attrs)])
    
            for i in range(len(self.split_list[attr_gid])):
                Temp_train_tuples[:,self.dataspace.pos_list[attr_gid][i]]= train_tuples[:,i]
                Temp_test_tuples[:,self.dataspace.pos_list[attr_gid][i]]= test_tuples[:,i]
            #print(Temp_train_tuples)
            Temp_train_tuples=self.dataspace.model.transform(pd.DataFrame(Temp_train_tuples,columns=self.dataspace.raw.columns))
            Temp_test_tuples=self.dataspace.model.transform(pd.DataFrame(Temp_test_tuples,columns=self.dataspace.raw.columns))
            
            
            index=0
            for i in self.split_list[attr_gid]:

                begin_index=0
                for j in self.dataspace.model.meta:
                    if j['name']==i:
                        break
                    else:
                        begin_index+=j['output_dimensions']
                print(begin_index)
                print(begin_index+j['output_dimensions'])
                if index==0:
                    final_train_tuples=Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]
                    final_test_tuples=Temp_test_tuples[:,begin_index:begin_index+j['output_dimensions']]
                else:
                    final_train_tuples=np.column_stack((final_train_tuples,Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]))
                    final_test_tuples=np.column_stack((final_test_tuples,Temp_test_tuples[:,begin_index:begin_index+j['output_dimensions']]))
                index+=1
                print(final_train_tuples.shape)
            print(final_train_tuples[0])
            print(final_test_tuples[0])
            
            return query_vector,final_train_tuples,final_test_tuples,train_labels,test_labels,train_rate,test_rate


        
        
 
        
    def  Generate_tasks(self,attr_gid):
        '''
        数据的组织方式：'query_id': {
        'querysapce_vector':[]
        'support_tuples':[[]] or None
        'support_labels':[]
        'query_tuples':[[]] or None
        'query_labels':[]
    
        }
        
        '''
        
        Task_dataset={}
        Task_dataset['train']={}
        Task_dataset['test']={}
       
        index=0
        while index<self.train_task_num:
            query_vector,train_tuples,test_tuples,train_labels,test_labels,train_rate,test_rate = self.Generate_single(attr_gid)
            
            if train_rate != None:
                if  0.05<=train_rate<=0.95 and  0.01<=test_rate<=0.99:
                    Task_dataset['train'][index]={}
                    Task_dataset['train'][index]['querysapce_vector']=query_vector.astype(float32)
                    Task_dataset['train'][index]['support_tuples']=train_tuples.astype(float32)
                    Task_dataset['train'][index]['support_labels']=train_labels.astype(int64)
                    Task_dataset['train'][index]['query_tuples']=test_tuples.astype(float32)
                    Task_dataset['train'][index]['query_labels']=test_labels.astype(int64)
                    index+=1
                    
             
                
        while index<self.train_task_num+self.test_task_num:
          query_vector,train_tuples,test_tuples,train_labels,test_labels,train_rate,test_rate = self.Generate_single(attr_gid)
          if train_rate != None:
              if  0.05<=train_rate<=0.95 and  0.01<=test_rate<=0.99:
                    Task_dataset['test'][index]={}
                    Task_dataset['test'][index]['querysapce_vector']=query_vector.astype(float32)
                    Task_dataset['test'][index]['support_tuples']=train_tuples.astype(float32)
                    Task_dataset['test'][index]['support_labels']=train_labels.astype(int64)
                    Task_dataset['test'][index]['query_tuples']=test_tuples.astype(float32)
                    Task_dataset['test'][index]['query_labels']=test_labels.astype(int64)
                    index+=1
                    
        return Task_dataset
                  
    def build(self):
        
        
        if not os.path.exists('{}/'.format(self.path)):
            os.mkdir('{}/'.format(self.path))
        
        for id in range(len(self.split_list)):
            if not os.path.exists('{}/attr_group_{}/'.format(self.path,id)):
                os.mkdir('{}/attr_group_{}/'.format(self.path,id))
            print("Group "+str(id)+" generate begin!")
            Task_dataset=self.Generate_tasks(id)
            print("Group "+str(id)+" generate over!")
     
            for i in Task_dataset['train'].keys():
                pickle.dump(Task_dataset['train'][i]['querysapce_vector'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_qv.p', 'wb'))
                pickle.dump(Task_dataset['train'][i]['support_tuples'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_s_tv.p', 'wb'))
                pickle.dump(Task_dataset['train'][i]['support_labels'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_s_y.p', 'wb'))
                pickle.dump(Task_dataset['train'][i]['query_tuples'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_q_tv.p', 'wb'))
                pickle.dump(Task_dataset['train'][i]['query_labels'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_q_y.p', 'wb'))
            for i in Task_dataset['test'].keys():
                pickle.dump(Task_dataset['test'][i]['querysapce_vector'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_qv.p', 'wb'))
                pickle.dump(Task_dataset['test'][i]['support_tuples'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_s_tv.p', 'wb'))
                pickle.dump(Task_dataset['test'][i]['support_labels'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_s_y.p', 'wb'))
                pickle.dump(Task_dataset['test'][i]['query_tuples'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_q_tv.p', 'wb'))
                pickle.dump(Task_dataset['test'][i]['query_labels'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_q_y.p', 'wb'))
          
      