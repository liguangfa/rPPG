import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import cv2
import random
from numpy import float16


'''directory_name='C:/Users/liguangfa/Desktop/subject/video_download2/'
path_list=os.listdir(directory_name)
#print('path_list:',path_list)'''
def get_snr0(area,i,image):
    im=[]
    q=[]
    for t in range(90):
        x = image[t][:, :, i]
        qt = x[image[t][:, :, i] > area]
        im.append(np.mean(x[image[t][:, :, i] > 20]))
        #q.append(np.sum(qt)/len(x[image[t][:, :, i] > 20]))
        q.append(np.mean(qt))
    #a=np.abs(im-area)
    a=np.max(im)-np.min(im)
    #q=np.abs(q)
    q=np.max(q)-np.min(q)
    p=np.mean(a)/np.mean(q)
    #print('a,q:',a,q,a/q)
    #print(0.0135*np.sqrt(p))
    return 0.0135*np.sqrt(p)

def load_data(batch,step,num_classes):#每次训练总项目数为batch*step

    while(True):

        heart_rate_low = 45
        heart_rate_high = 99
        heart_rate_resolution = 2
        length_vid = 64
        img_width = 64
        img_height = 64
        sampling = 1 / 15
        t = np.linspace(0, length_vid * sampling - sampling, length_vid)
        #no_classes = len(heart_rates)  # 60
        #print('freqs:',freqs)
        #labels = np.arange(0, no_classes + 1, 1, dtype='long')
        #labels_cat = labels
        min = (-3, -1, -1)
        max = (3, 1, 1)
        coeff = {
            'a0': 0.440240602542388,
            'a1': -0.334501803331783,
            'b1': -0.198990393984879,
            'a2': -0.050159136439220,
            'b2': 0.099347477830878,
            'w': 2 * np.pi
        }
        length=len(t)
        for count in range(0,step):#(0,batch*step,,num_class),(0,5000,18),表示在step级
            heart_rates = np.arange(heart_rate_low, heart_rate_high, heart_rate_resolution)  # 40-100
            freqs = heart_rates / 60
            images=[]
            images_label=[]
            num= []
            le=[]
            l=[]
            freqs=random.sample(list(freqs),batch)
            count=count+1
            #print('count-1:', count)
            #print('freqs:', freqs)
            #print('freqs:', np.array(freqs)*60)
            for i, freq in enumerate(freqs):#在batch级别
                #print('i,freqs:',i,freq)
                #print('count-2:',count)
                t2 = t + (np.random.randint(low=0, high=33) * sampling)
                #print('t2:',t2)
                signal = (coeff['a0'] + coeff['a1'] * np.cos(t2 * coeff['w'] * freq)
                          + coeff['b1'] * np.sin(t2 * coeff['w'] * freq)
                          + coeff['a2'] * np.cos(2 * t2 * coeff['w'] * freq)
                          + coeff['b2'] * np.sin(2 * t2 * coeff['w'] * freq))

                signal = signal - np.min(signal)
                signal = signal / (np.max(signal)+0.3)
                tend_list=[-10,-8,-4,-1,1,4,8,10]
                a=random.sample(tend_list,1)
                tend = np.linspace(0, 1, length)
                tend = tend * tend
                tend = tend - min[2]
                tend = 0.5 * max[2] * tend / np.max(tend)

                signal = np.expand_dims(signal+tend*a, 1)
                signal = (signal - np.min(signal))/100

                x=[]
                label=int(60*freq)
                #print('label:',label)
                n=[l for l in range(1,58)]#[l for l in range(1,12)]
                m=random.sample(n,1)
                m=m[0]
                directory = 'C:/Users/liguangfa/Desktop/subject/processed_train2/'#'C:/Users/liguangfa/Desktop/subject/processed_test/'
                emtion_video = h5py.File(directory + str(m) + '/map.h5')
                emtion_video = emtion_video['map.h5']
                #imagexa=emtion_video[0]
                #am=[o for o in range(125,165,3)]
                #amp=random.sample(am,1)[0]
                #print('average_pixel:',m, label,int(np.mean(emtion_video[0,:,:,0][emtion_video[0,:,:,0]>10])),int(np.mean(emtion_video[0,:,:,1][emtion_video[0,:,:,1]>10])),int(np.mean(emtion_video[0,:,:,2][emtion_video[0,:,:,2]>10])))
                ave=random.randint(10,15)

                area_0=np.mean(emtion_video[0,:,:,0][emtion_video[0,:,:,0]>0.9*ave])
                area_1 = np.mean(emtion_video[0,:, :, 1][emtion_video[0,:, :, 1] > 1.2*ave])
                area_2 = np.mean(emtion_video[0,:, :, 2][emtion_video[0,:, :, 2] > 1.4*ave])

                signal = (signal) / (np.max(signal) - np.min(signal))

                p1 = get_snr0(area_1, 1, emtion_video)
                p0 = 0.66 * p1  # get_snr0(area_0, 0,emtion_video )
                p2 = p0#get_snr0(area_2, 2, emtion_video)

                for k in range(len(signal)):

                    #imagea = cv2.resize(imagexa, (64, 64),interpolation=cv2.INTER_AREA)
                    image=emtion_video[k]

                    x0 = image[:,:,0]
                    x1 = image[:, :, 1]
                    x2 = image[:, :, 2]

                    area = np.random.uniform(low=1.01, high=1.15)
                    area0 = area * area_0  # 原来是乘以1.15
                    area1 = area * area_1
                    area2 = area * area_2
                    #amplitude = np.random.uniform(low=0.995, high=1.01)
                    amplitude1 = np.random.uniform(low=0.995, high=1.01)
                    amplitude2 = np.random.uniform(low=0.998, high=1.002)

                    af0 = 0.025 * signal[k]#0.007
                    af1 = 0.05 * signal[k]#0.014
                    af2 = 0.025* signal[k]

                    '''x0[image[:,:,0]>area0]=(x0[image[:,:,0]>area0]+signal[k][0]*amp)* amplitude
                    x1[image[:,:,1] > area1] = (x1[image[:,:,1] > area1]  +signal[k][0] * amp) * amplitude
                    x2[image[:,:,2] > area2] = (x2[image[:,:,2] > area2] +signal[k][0] * amp) * amplitude'''

                    q0 = x0[image[:, :, 0] > area0]
                    q1 = x1[image[:, :, 1] > area1]
                    q2 = x2[image[:, :, 2] > area2]
                    #print('range0:', np.max(q0), np.min(q0), len(q0))
                    #print('range1:', np.max(q1), np.min(q1), len(q1))
                    #print('range2:', np.max(q2), np.min(q2), len(q2))
                    x0[image[:, :, 0] > area0] = q0 + af0 * q0*amplitude1
                    x1[image[:, :, 1] > area1] = q1 + af1 * q1*amplitude2
                    x2[image[:, :, 2] > area2] = q2 + af2 * q2*amplitude1
                    '''x0[image[:, :, 0] > area0] = np.squeeze(np.array([o * (af + 1) for o in q0])) * amplitude1
                    x1[image[:, :, 1] > area1] = np.squeeze(np.array([o * (af + 1) for o in q1])) * amplitude2
                    x2[image[:, :, 2] > area2] = np.squeeze(np.array([o * (af + 1) for o in q2])) * amplitude1'''

                    image[:, :, 0]=x0
                    image[:, :, 1]=x1
                    image[:, :, 2]=x2

                    #add noise
                    '''amplitude = np.random.uniform(low=33, high=33.1)  # (low=1.5, high=4)
                    noise_energy = amplitude * 3.6 * np.random.uniform(low=1, high=6) / 60
                    image=image+np.random.normal(size=(img_height, img_height, 3), loc=0.5, scale=0.25)* noise_energy'''

                    #cv2.imshow('video', image.astype('uint8'))
                    #cv2.waitKey(10)
                    x.append(image.astype('uint8'))

                pulse = [0 for i in range(num_classes)]
                index = (int(label) - heart_rate_low) // 2
                if index > (num_classes - 1):
                    index = num_classes - 1
                pulse[index] = 1
                images_label.append(pulse)

                x=np.array(x)/255
                images.append(x.astype(float16))
                num.append(index)

                u = index
                sig = math.sqrt(0.5)  # 标准差δ,决定曲线胖瘦
                x = np.linspace(1, num_classes, num=num_classes)  # 定义域
                leng = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
                le.append(leng)

                #l2 = np.random.rand(images.shape[0], 1)
                #l.append(l2)
                #le = np.array(le)

            images=np.array(images)
            images_label=np.array(images_label)
            numclass=np.array(num)
            le=np.array(le)
            #l=np.array(l)
            #print('numshape:',numclass)

            yield ({'input_1': images},
               {'new_dense': images_label})#,'input_2':numclass,,'dense_1':images_label


def load_test(batch,step,num_classes):#每次训练总项目数为batch*step

    while(True):

        heart_rate_low = 45
        heart_rate_high = 81
        heart_rate_resolution = 2
        length_vid = 90
        img_width = 64
        img_height = 64
        sampling = 1 / 15
        t = np.linspace(0, length_vid * sampling - sampling, length_vid)
        #no_classes = len(heart_rates)  # 60
        #print('freqs:',freqs)
        #labels = np.arange(0, no_classes + 1, 1, dtype='long')
        #labels_cat = labels
        min = (-3, -1, -1)
        max = (3, 1, 1)
        coeff = {
            'a0': 0.440240602542388,
            'a1': -0.334501803331783,
            'b1': -0.198990393984879,
            'a2': -0.050159136439220,
            'b2': 0.099347477830878,
            'w': 2 * np.pi
        }
        length=len(t)
        for count in range(0,step):#(0,batch*step,,num_class),(0,5000,18),表示在step级
            heart_rates = np.arange(heart_rate_low, heart_rate_high, heart_rate_resolution)  # 40-100
            freqs = heart_rates / 60
            images=[]
            images_label=[]
            num_class = []
            freqs=random.sample(list(freqs),batch)
            count=count+1
            #print('count-1:', count)
            #print('freqs:', freqs)
            #print('freqs:', np.array(freqs)*60)
            for i, freq in enumerate(freqs):#在batch级别
                #print('i,freqs:',i,freq)
                #print('count-2:',count)
                t2 = t + (np.random.randint(low=0, high=60) * sampling)
                fit = random.randint(5, 85)
                t2_1 = t2[0:fit]
                t2_2 = t2[fit:]
                signal = []
                ap = random.randint(95, 105) / 100
                signal1 = (coeff['a0'] + coeff['a1'] * np.cos(t2_1 * coeff['w'] * freq)  # 0.97-1.03
                           + coeff['b1'] * np.sin(t2_1 * coeff['w'] * freq)  # 0.98-1.08
                           + coeff['a2'] * np.cos(2 * t2_1 * coeff['w'] * freq)  # 0.95-1.05
                           + coeff['b2'] * np.sin(2 * t2_1 * coeff['w'] * freq))
                freq1 = ap * freq
                signal2 = (coeff['a0'] + coeff['a1'] * np.cos(t2_2 * coeff['w'] * freq1)  # 0.97-1.03
                           + coeff['b1'] * np.sin(t2_2 * coeff['w'] * freq1)  # 0.98-1.08
                           + coeff['a2'] * np.cos(2 * t2_2 * coeff['w'] * freq1)  # 0.95-1.05
                           + coeff['b2'] * np.sin(2 * t2_2 * coeff['w'] * freq1))
                signal.extend(signal1)
                signal.extend(signal2)

                signal = signal - np.min(signal)
                signal = signal / (np.max(signal)+0.3)
                tend_list=[-8,-6,-4,-1,1,4,6,8]
                a=random.sample(tend_list,1)
                tend = np.linspace(0, 1, length)
                tend = tend * tend
                tend = tend - min[2]
                tend = 0.5 * max[2] * tend / np.max(tend)

                signal = np.expand_dims(signal+tend*a, 1)
                signal = (signal - np.min(signal))/100

                x=[]
                label=int(60*freq)
                #print('label:',label)
                n=[l for l in range(1,58)]#[l for l in range(1,12)]
                m=random.sample(n,1)
                m=m[0]
                directory = 'C:/Users/liguangfa/Desktop/subject/processed_train2/'#'C:/Users/liguangfa/Desktop/subject/processed_test/'
                emtion_video = h5py.File(directory + str(m) + '/map.h5')
                emtion_video = emtion_video['map.h5']
                imagexa=emtion_video[0]
                am=[o for o in range(120,155,3)]
                amp=random.sample(am,1)[0]
                #print('average_pixel:',m, label,int(np.mean(emtion_video[0,:,:,0][emtion_video[0,:,:,0]>10])),int(np.mean(emtion_video[0,:,:,1][emtion_video[0,:,:,1]>10])),int(np.mean(emtion_video[0,:,:,2][emtion_video[0,:,:,2]>10])))

                ave=random.randint(10,15)

                area_0=np.mean(emtion_video[0,:,:,0][emtion_video[0,:,:,0]>0.9*ave])
                area_1 = np.mean(emtion_video[0,:, :, 1][emtion_video[0,:, :, 1] > 1.2*ave])
                area_2 = np.mean(emtion_video[0,:, :, 2][emtion_video[0,:, :, 2] > 1.4*ave])

                for k in range(len(signal)):

                    imagea = cv2.resize(imagexa, (64, 64),interpolation=cv2.INTER_AREA)
                    image=emtion_video[k]

                    x0=image[:,:,0]
                    x1 = image[:, :, 1]
                    x2 = image[:, :, 2]

                    area = np.random.uniform(low=1.1, high=1.35)
                    area0 = area * area_0  # 原来是乘以1.15
                    area1 = area * area_1
                    area2 = area * area_2

                    amplitude = np.random.uniform(low=0.995, high=1.01)
                    x0[image[:,:,0]>area0]=(x0[image[:,:,0]>area0]+signal[k][0]*amp)* amplitude
                    x1[image[:,:,1] > area1] = (x1[image[:,:,1] > area1]  +signal[k][0] * amp) * amplitude
                    x2[image[:,:,2] > area2] = (x2[image[:,:,2] > area2] +signal[k][0] * amp) * amplitude

                    image[:, :, 0]=x0
                    image[:, :, 1]=x1
                    image[:, :, 2]=x2

                    '''amplitude = np.random.uniform(low=33, high=33.1)  # (low=1.5, high=4)
                    noise_energy = amplitude * 3.6 * np.random.uniform(low=1, high=6) / 20
                    image=image+np.random.normal(size=(img_height, img_height, 3), loc=0.5, scale=0.25)* noise_energy'''

                    #cv2.imshow('video', image.astype('uint8'))
                    #cv2.waitKey(40)
                    x.append(image.astype('uint8'))

                pulse = [0 for i in range(num_classes)]
                index = (int(label) - heart_rate_low) // 2
                if index > (num_classes - 1):
                    index = num_classes - 1
                pulse[index] = 1
                images_label.append(pulse)

                x=np.array(x)/255
                images.append(x.astype(float16))

            images=np.array(images)
            images_label=np.array(images_label)

            yield ({'input_1': images},
               {'dense': images_label})


if __name__ == '__main__':
    load_data(batch=4,step=10,num_classes=18)


