import h5py
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from sklearn.model_selection import KFold,train_test_split

# # 转换tsv文件为hdf5文件


# os.makedirs('./norm',exist_ok=True)

# # def norm(input_data):
# #     mean=np.mean(input_data,axis=0,dtype=np.float32)
# #     print(mean.shape)
# #     std=np.std(input_data,axis=0,dtype=np.float32)
# #     return (input_data-mean)/std

def norm(scaler,input_data):
        # maximums,minimums,avgs = input_data.max(axis = 0), input_data.min(axis = 0),input_data.sum(axis = 0)/input_data.shape[0]
        # #归一化
        # norm_1=(input_data-minimums)/(maximums-minimums)
        input_data=scaler.transform(input_data)
        print('11',input_data.max())
        print('11',input_data.min())
        #标准化
        #mean=np.mean(norm_1,axis=0)
        #std=np.std(norm_1,axis=0)

        #print('22',mean)
        #print(len(mean))
        #print('22',std)
        
        # print(maximums.shape)
        # np.save(save_norm+'/max.npy',maximums)
        # np.save(save_norm+'/min.npy',minimums)
        # np.save(save_norm+'/avg.npy',avgs)
        return (input_data-0.5)/0.5

data = pd.read_csv('./data_new/GDC_PANCANCER.htseq_fpkm-uq_finalreal.tsv', sep='\t', index_col=0, chunksize=10000)

chunk_list = []
for chunk in data:
    chunk_list.append(chunk.values.astype(np.float32))
# 19760,9896
data = np.concatenate(chunk_list, axis=0)
data=data.T
# 9896,19760
print(data.shape)
print(np.max(data))
print(np.min(data))
# data=norm(data)
print(np.max(data))
print(np.min(data))
# split
data_train,data_test=train_test_split(data,test_size=0.1,random_state=0)
print('len',data_test.shape[0])
print('len',data_train.shape[0])

scaler=preprocessing.MinMaxScaler()
scaler.fit(data)
data_train=norm(scaler,data_train)
data_test=norm(scaler,data_test)
# data_train=scaler.transform(data_train)
# data_test=scaler.transform(data_test)

# avgs,maximums,minimums,data_train=norm(data_train)
# data_test=(data_test-avgs)/(maximums-minimums)



# data=norm(data)
print(np.max(data_train))
print(np.min(data_test))
print('converting...')
data_file=h5py.File("./data_new/GDC_PANCANCER.htseq_fpkm-uq_final.hdf5","w")
# g_gene=data_file.create_group('Ensembl_id')
# g_gene.create_dataset("ENSG",data=gene_id)
length=data_file.create_group('dataset_dim')
length.create_dataset('train',data=data_train.shape)
length.create_dataset('test',data=data_test.shape)

g=data_file.create_group('pancancer_exp')
# print(data[1,:].shape)
# print(np.max(data[2,:]))
# print(data[1,:])
index=0
for i in range(data_train.shape[0]):
    g.create_dataset('train_%d'%i,data=data_train[i,:])
    index+=1

index2=0
for j in range(data_test.shape[0]):
    g.create_dataset('test_%d'%j,data=data_test[j,:])
    index2+=1

print('finished!')
print('total is',index)
print('total test is',index2)
print(len(g.keys()))
data_file.close()
print()
print()


mirna_data = pd.read_csv('./data_new/GDC_PANCANCER.mirna_finalreal.tsv', sep='\t', index_col=0)

# chunk_list = []
# for chunk in data:
#     chunk_list.append(chunk.values.astype(np.float32))
# # 19760,9896
# data = np.concatenate(chunk_list, axis=0)
mirna_data=mirna_data.values.astype(np.float32).T
# 9896,19760
print(mirna_data.shape)
print(np.max(mirna_data))
print(np.min(mirna_data))
# data=norm(data)
print(np.max(mirna_data))
print(np.min(mirna_data))
# split
midata_train,midata_test=train_test_split(mirna_data,test_size=0.1,random_state=0)
print('len',midata_test.shape[0])
print('len',midata_train.shape[0])

scaler=preprocessing.MinMaxScaler()
scaler.fit(mirna_data)
midata_train=norm(scaler,midata_train)
midata_test=norm(scaler,midata_test)
# data_train=scaler.transform(data_train)
# data_test=scaler.transform(data_test)

# avgs,maximums,minimums,data_train=norm(data_train)
# data_test=(data_test-avgs)/(maximums-minimums)



# data=norm(data)
# print(np.max(data_train))
# print(np.min(data_test))
print('converting...')
data_file=h5py.File("./data_new/GDC_PANCANCER.mirna_final.hdf5","w")
# g_gene=data_file.create_group('Ensembl_id')
# g_gene.create_dataset("ENSG",data=gene_id)
length=data_file.create_group('dataset_dim')
length.create_dataset('train',data=midata_train.shape)
length.create_dataset('test',data=midata_test.shape)

g=data_file.create_group('pancancer_exp')
# print(data[1,:].shape)
# print(np.max(data[2,:]))
# print(data[1,:])
index=0
for i in range(midata_train.shape[0]):
    g.create_dataset('train_%d'%i,data=midata_train[i,:])
    index+=1

index2=0
for j in range(midata_test.shape[0]):
    g.create_dataset('test_%d'%j,data=midata_test[j,:])
    index2+=1

print('finished!')
print('total is',index)
print('total test is',index2)
print(len(g.keys()))
data_file.close()
print()
print()





print("convert cancer type....")

for data_type in ["TCGA-BLCA","TCGA-BRCA","TCGA-KIRC","TCGA-HNSC","TCGA-LGG","TCGA-LIHC","TCGA-LUAD","TCGA-LUSC","TCGA-OV","TCGA-STAD","TCGA-COAD","TCGA-SARC","TCGA-UCEC","TCGA-CESC","TCGA-PRAD","TCGA-SKCM"]:
    data=pd.read_csv(os.path.join('data_new','%s.htseq_fpkm-uq_finalreal.tsv' % data_type),sep='\t',index_col=0)
    mi_data=pd.read_csv(os.path.join('data_new','%s.mirna_final.tsv' % data_type),sep='\t',index_col=0)
    # data.head()
    clinical_data=pd.read_csv(os.path.join('data_new','%s.survival_clean.tsv' % data_type),sep='\t',index_col=0)

    gene_id=np.array(data.index,dtype=object)
    print('%s:'%data_type)
    print(gene_id)
    print(gene_id.shape)
    data=data.values.astype(np.float32).T
    print("pre")
    print(data.max())
    print(data.min())
    scaler_mrna=preprocessing.MinMaxScaler()
    scaler_mrna.fit(data)
    data=norm(scaler_mrna,data)
    print(data.shape)
    print("end")
    print(data.max())
    print(data.min())

    mi_gene_id=np.array(mi_data.index,dtype=object)
    print('%s:'%data_type)
    print(mi_gene_id)
    print(mi_gene_id.shape)
    mi_data=mi_data.values.astype(np.float32).T
    print("pre")
    print(data.max())
    print(data.min())
    scaler_mirna=preprocessing.MinMaxScaler()
    scaler_mirna.fit(mi_data)
    mi_data=norm(scaler_mirna,mi_data)
    print("end")
    print(data.max())
    print(data.min())
    
    print(mi_data.shape)

    
    os_event=clinical_data['OS'].values.astype(np.int32).reshape((-1,1))
    os_time=clinical_data['OS.time'].values.astype(np.int32).reshape((-1,1))
    # print('normalizing....')

    # data=norm(data)
    # print(np.max(data))
    # print(np.min(data))

    if os_event.shape[0] == os_time.shape[0] and os_event.shape[0]==data.shape[0] and os_event.shape[0]==mi_data.shape[0]:
        exp_data=np.concatenate((data,mi_data,os_event,os_time),axis=1)
        print(exp_data.shape)
    else:
        raise NotImplementedError("error")

    data_file=h5py.File('./data_new/%s.5_folds.hdf5'%data_type,'w')
    string_dt = h5py.special_dtype(vlen=str)
    g_gene=data_file.create_group('exp_name')
    g_gene.create_dataset('ENSG',data=gene_id,dtype=string_dt)
    g_gene.create_dataset('mi-id',data=mi_gene_id,dtype=string_dt)
    g=data_file.create_group('exp')
    print('exp_data',exp_data.shape)
    # print(exp_data.max())
    # print(exp_data.min())
    # # exp_data=norm(exp_data)
    # print(exp_data.max())
    # print(exp_data.min())

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for idx,(train_index,test_index) in enumerate(kf.split(exp_data)):
        print(idx)
        f=g.create_group('cross_%d'%idx)

        train_data=exp_data[train_index,:]
        test_data=exp_data[test_index,:]
        # norm
        # print(train_data.shape)
        # print(test_data.shape)
        # # print('norm')
        # scaler=preprocessing.StandardScaler()
        # scaler.fit(train_data)
        # train_data=scaler.transform(train_data)
        # test_data=scaler.transform(test_data)
        # avgs,maximums,minimums,data_train=norm(data_train)
        # data_test=(data_test-avgs)/(maximums-minimums)


        # print(train_data.shape)
        # print(test_data.shape)
        f.create_dataset('train',data=train_data)
        f.create_dataset('test',data=test_data)

    print('finished!')
    data_file.close()
    print()
    print()
