import numpy as np
import scipy.io
import copy
#generating STFT feature
def get_stft_data(path = 'SE_STFT.mat'):
    data = scipy.io.loadmat(path)
    data = np.array(data['DB'])
    tags = data[:, 0]
    tags.astype(int)
    data = data[:, 1:]
    newdata = np.zeros([len(data), 25, 13])

    for i in range(len(data)):
        newdata[i] = copy.deepcopy(data[i].reshape([25,13]))
    print('=============Successfully generate CNN data==========')
    return tags, newdata
#generating MFCC feature
def get_data(path = 'SE_MFCC.mat'):
    data = scipy.io.loadmat(path)
    # key = path.split('_')[2][0:3]
    data = np.array(data['DB'])
    tags = data[:, 0]
    tags.astype(int)
    data = data[:, 1:]
    newdata = np.zeros([len(data), 25, 13])

    for i in range(len(data)):
        newdata[i] = copy.deepcopy(data[i].reshape([13, 25]).T)
    newdata.reshape([-1, 325])
    print('=============Successfully generate CNN data==========')
    return  tags, newdata
#generating mixed feature
def get_moredata(path='SE_MFCC.mat'):
    data = scipy.io.loadmat(path)
    stft = scipy.io.loadmat('SE_STFT.mat')
    data = np.array(data['DB'])
    stft = np.array(stft['DB'])
    tags = data[:, 0]
    tags.astype(int)
    data = data[:, 1:]
    stft = stft[:, 1:]
    newdata = np.zeros([len(data), 25, 13])
    stft_data= np.zeros([len(stft), 25, 13])
    hybrid_data = np.zeros([len(stft), 25, 26])
    for i in range(len(data)):
        newdata[i] = copy.deepcopy(data[i].reshape([13, 25]).T)
    for i in range(len(stft)):
        stft_data[i] = copy.deepcopy(stft[i].reshape([25, 13]))
    for i in range(len(newdata)):
        a= newdata[i]
        b=stft_data[i]
        c = np.hstack((a, b))
        hybrid_data[i] = c

    print('=============Successfully generate CNN data==========')
    return tags, hybrid_data
#calculate the differential data(the difference between two adjunct vectors)
def differential_data(path = 'SE_MFCC.mat'):
    data = scipy.io.loadmat(path)
    #key = path.split('_')[2][0:3]
    data = np.array(data['DB'])
    tags = data[:,0]
    tags.astype(int)
    data = data[:,1:]
    newdata = np.zeros([len(data), 25, 13])


    for i in range(len(data)):
        newdata[i] = copy.deepcopy(data[i].reshape([13, 25]).T)
    diff_data = np.zeros([len(data), 312])
    for i in range(len(data)):
        for j in range(len(newdata[0])-1):
            newdata[i][j] = (copy.deepcopy(newdata[i][j+1] - newdata[i][j]))
        diff_data[i] = newdata[i][:24].reshape([312])
    for i in range(len(data)):
        diff_data[i] = copy.deepcopy(diff_data[i].reshape([312]))
    print('=============Successfully generate differential data==========')
    return tags,diff_data
#generate unidimentional MFCC data for ANN
def DNN_get_data(path = 'SE_MFCC.mat'):
    data = scipy.io.loadmat(path)
    #key = path.split('_')[2][0:3]
    data = np.array(data['DB'])
    tags = data[:,0]
    tags.astype(int)
    data = data[:,1:]
    return tags,data
#generate unidimentional STFT data for ANN
def DNN_get_stft_data(path = 'SE_STFT.mat'):
    data = scipy.io.loadmat(path)
    #key = path.split('_')[2][0:3]
    data = np.array(data['DB'])
    tags = data[:,0]
    tags.astype(int)
    data = data[:,1:]
    return tags,data
#generate mixed unidimentional data for ANN
def load_extra_data(path='SE_MFCC.mat'):
    data = scipy.io.loadmat(path)
    STFT_DATA = scipy.io.loadmat('SE_STFT.mat')
    data = np.array(data['DB'])
    STFT_DATA = np.array(STFT_DATA['DB'])
    tags = data[:,0]
    tags.astype(int)
    stft_tags = STFT_DATA[:,0]
    stft_data = STFT_DATA[:,1:]
    data = data[:,1:]
    data = np.hstack((data, stft_data))
    return tags,data
# def load_diff_MFCC_data(path='SE_MFCC.mat'):
#     data = scipy.io.loadmat(path)
#     STFT_DATA = scipy.io.loadmat('SE_STFT.mat')
#     data = np.array(data['DB'])
#     STFT_DATA = np.array(STFT_DATA['DB'])
#     tags = data[:,0]
#     tags.astype(int)
#     stft_tags = STFT_DATA[:,0]
#     stft_data = STFT_DATA[:,1:]
#     _, diff_data = differential_data()
#     data = data[:,1:]
#     data = np.hstack((data, diff_data))
#     return tags,data
