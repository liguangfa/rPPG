import numpy as np
from random import shuffle
import h5py
import os
import math
import tensorflow as tf
from numpy import float16
from scipy.signal import find_peaks

def generate_filename(image_path):
    directory_name = image_path
    path_list = os.listdir(image_path)
    #print(path_list)
    
    # path_list.remove('.DS_Store')
    path_list = [int(float(x)) for x in path_list]
    path_list.sort()
    
    #shuffle(path_list)
    #print(path_list)
    directory_name = list(map(str, path_list))
    directory_name=[image_path+x for x in directory_name]
    #directory_name_map=[image_path+x+'/map.h5' for x in directory_name]
    #directory_name_bpm = [image_path + x + '/bpm.h5' for x in directory_name]
    #print(directory_name)

    return directory_name

def read_data(filename,batch_size):#,batchsize
    batches = (len(filename) + batch_size - 1) // batch_size
    #print("filename:",filename)
    #shuffle(filename)
    #filename=filename
    #print(filename)
    while (True):
      shuffle(filename)      
      for i in range(batches):
        images=[]
        #labels=[]
        num_class=[]
        num=[]
        le=[]
        X = filename[i * batch_size: (i + 1) * batch_size]#x是每批的数量
        for path in X:
          #print('path:',path)
          label = h5py.File(path+'/bpm.h5')
          image = h5py.File(path+'/map.h5')
          image=image['map.h5'] #尺寸为(64,128,128,3)
          label=label['bpm.h5']

          label=np.array(label)

          image=np.array(image)/255
          images.append(image.astype(float16))
          pulse = [0 for i in range(18)]
          index = (int(label) - 45) // 2
          if index > 17:
              index = 17
          pulse[index] = 1
          num_class.append(pulse)
          num.append(index)

          u = index
          sig = math.sqrt(1)  # 标准差δ,决定曲线胖瘦
          x = np.linspace(1, 18, num=18)  # 定义域
          leng = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
          le.append(leng)

        num_class=np.array(num_class,dtype=float16)
        images = np.array(images,dtype=float16) # 这句话解决了ValueError: Argument must be a dense tensor:
        label = np.array(num)
        l2 = np.random.rand(images.shape[0], 1)
        le=np.array(le)


        
        yield ({'input_1':images},
            {'dense':num_class})#

def read_datavar(filenameva):
    
    images=[]
    labels=[]
    waves=[]
    images132 = []
    labels132 = []
    images232 = []
    labels232 = []
    num_class=[]
    for path in filenameva:
      label = h5py.File(path+'/bpm.h5')
      image = h5py.File(path+'/map.h5')
      wave=h5py.File(path+'/rppg.h5')
      wave=wave['rppg.h5']
      image=image['map.h5'] #尺寸为(64,128,128,3)
      label=label['bpm.h5']
      #z=np.array([2.8 for x in label])
      #print('z:',z)
      #label=(label+z)/6.5
      
      
      #label=(label-min(label))/(max(label)-min(label)) #label中的数据变成0-1
      #print('min_label:',min_label)
      #print("max(label)-min(label)",max(label)-min(label))
      #print('label:',label)

      '''image132 = [image[i] for i in range(0,64,2)]
      image232 = [image[i+1] for i in range(0,64,2)]
      label132 = [label[i]/127 for i in range(0,64,2)]
      label232 = [label[i + 1] for i in range(0,64,2)]'''

      #label = (label - np.mean(label)) / np.std(label)          
      #label132 = (label132 - np.mean(label132)) / np.std(label132)
      #label232 = (label232 - np.mean(label232)) / np.std(label232)
      '''image=np.array(image)/255
      image132=np.array(image132)/255
      image232 = np.array(image232) / 255'''
      label=np.mean(label)

      label=np.array(label)
      image=np.array(image)
      images.append(image)
      labels.append(label.astype(float16))
      wave = np.array(wave)
      waves.append(wave)
      '''images132.append(image132)
      images232.append(image232)
      labels132.append(label132)
      labels232.append(label232)'''
      #print('heart:',label)
      '''pulse=[0 for i in range(60)]
      if label>=100:
        label=99
      #print('heart_reart:',label)
      pulse[int(label)-40]=1
      num_class.append(pulse)'''

      pulse=[0 for i in range(25)]
      index=(int(label)-45)//2

      #print('index:', index)
      if index > 24:
          index = 24
      #print('index:', index)

      pulse[index] = 1
      #print('pulse:',pulse)
      '''if label == 45:
          pulse = [1, 0, 0]
      elif label == 47:
          pulse = [0, 1, 0]
      elif label == 49:
          pulse = [0, 0, 1]'''

      num_class.append(pulse)


      '''x=label
      xpeak_id, xpeak_property = find_peaks(x, distance=18, prominence=0.11)  # ,prominence=0.1
      #print('xpeak_id:', xpeak_id)
      xinter = []
      if len(xpeak_id)==1:
        xfreq=42
      else:
        for i in range(1, min([len(xpeak_id)])):
          xper_inter = xpeak_id[i] - xpeak_id[i - 1]
          xinter.append(xper_inter)
        xfreq = 1800 / np.mean(xinter)
      #print('xfreq:',xfreq)
      pulse=[0 for i in range(60)]
      pulse[int(xfreq)-40]=1
      num_class.append(pulse)'''

    num_class=np.array(num_class,dtype=float16)
    images = np.array(images,dtype=float16)  # 这句话解决了ValueError: Argument must be a dense tensor:
    waves = np.array(waves, dtype=float16)
    labels = np.array(labels)
    labels = np.array(labels)
    '''images132=np.array(images132,dtype=float16)
    labels132=np.array(labels132)
    images232=np.array(images232,dtype=float16)
    labels232=np.array(labels232)'''
    #print('labels:',labels.shape)

    #data_train=[images,images132,images232,labels,labels132,labels232,num_class]
    data_train=[images,labels,waves,num_class]
        
    return data_train


def read_datava(filenameva):
    images = []
    labels = []
    waves = []
    images132 = []
    labels132 = []
    images232 = []
    labels232 = []
    num_class = []
    for path in filenameva:
        label = h5py.File(path + '/bpm.h5')
        image = h5py.File(path + '/map.h5')
        #wave = h5py.File(path + '/rppg.h5')
        #wave = wave['rppg.h5']
        image = image['map.h5']  # 尺寸为(64,128,128,3)
        label = label['bpm.h5']

        label = np.mean(label)

        label = np.array(label)
        image = np.array(image)
        images.append(image)
        labels.append(label.astype(float16))
        #wave = np.array(wave)
        #waves.append(wave)


        pulse = [0 for i in range(25)]
        index = (int(label) - 45) // 2
        if index > 24:
            index =24
        pulse[index] = 1
        num_class.append(pulse)



    num_class = np.array(num_class, dtype=float16)
    images = np.array(images, dtype=float16)  # 这句话解决了ValueError: Argument must be a dense tensor:
    waves = np.array(waves, dtype=float16)
    labels = np.array(labels)
    labels = np.array(labels)

    data_train = [images, labels, num_class]

    return data_train

def read_ppg(filenameva):
    images = []
    labels = []
    waves = []
    images132 = []
    labels132 = []
    images232 = []
    labels232 = []
    num_class = []
    for path in filenameva:
        label = h5py.File(path + '/bpm.h5')
        image = h5py.File(path + '/map.h5')
        #wave = h5py.File(path + '/rppg.h5')
        #wave = wave['rppg.h5']
        image = image['map.h5']  # 尺寸为(64,128,128,3)
        label = label['bpm.h5']
        #label = np.mean(label)
        label = np.array(label)
        image = np.array(image)
        images.append(image)
        labels.append(label.astype(float16))

    #num_class = np.array(num_class, dtype=float16)
    images = np.array(images, dtype=float16)  # 这句话解决了ValueError: Argument must be a dense tensor:
    waves = np.array(waves, dtype=float16)
    labels = np.array(labels)
    #print('labels:',labels)

    data_train = [images, labels]

    return data_train

if __name__ == '__main__':
    '''#generate_filename(image_path='G:/rppgnet_train/train2_h5/')
    #read_data()'''