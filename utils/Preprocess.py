import numpy as np
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from operator import truediv
import torch

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels=False):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.float32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def feature_normalize(data):
    mu = torch.mean(data,dim=0)
    std = torch.std(data,dim=0)
    return torch.div((data - mu),std)

def L2_Norm(data):
    norm=np.linalg.norm(data, ord=2)
    return truediv(data,norm)
    
def feature_normalize2(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return truediv((data - mu),std)

def Preprocess1(XPath,yPath,dataset, Windowsize=25, Patch_channel=15):

    # X, y = loadData1(dataset)

    # X, pca = applyPCA(X, numComponents=Patch_channel)

    # X, y = createImageCubes(X, y, windowSize=Windowsize)
    # X=feature_normalize2(X)
    # np.save(XPath, X)
    # np.save(yPath,y)

    X, y = loadData1(dataset)
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    X = feature_normalize2(X)
    y = y.reshape(-1)
    np.save(XPath, X)
    np.save(yPath, y)

    return 0

def Preprocess2(XPath,yPath,dataset, Windowsize=25, Patch_channel=15):

    # X, y = loadData2(dataset)

    # X, pca = applyPCA(X, numComponents=Patch_channel)

    # X, y = createImageCubes(X, y, windowSize=Windowsize)
    # X=feature_normalize2(X)
    # np.save(XPath, X)
    # np.save(yPath,y)

    X, y = loadData2(dataset)
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    X = feature_normalize2(X)
    y = y.reshape(-1)
    np.save(XPath, X)
    np.save(yPath, y)

    return 0

def loadData1(name):
    data_path = os.path.join(os.getcwd(), 'detectiondatasets')
    if name == 'Sandiego2':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego2', 'agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego2','groundtruth.mat'))['gt']
    elif name == 'Sandiego100':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'groundtruth.mat'))['groundtruth']
    elif name == 'MUUFL':
        data = sio.loadmat(os.path.join(data_path, 'MUUFL', 'agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'MUUFL', 'groundtruth.mat'))['groundtruth']
    elif name == 'GF5':
        data = sio.loadmat(os.path.join(data_path, 'GF5', 'agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'GF5', 'groundtruth.mat'))['groundtruth']
    elif name == 'Sandiego':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego', 'agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego', 'groundtruth.mat'))['groundtruth']
    elif name == 'Synthetic':
        data = sio.loadmat(os.path.join(data_path, 'Synthetic', 'agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Synthetic', 'groundtruth.mat'))['groundtruth']
    elif name == 'HYDICE':
        data = sio.loadmat(os.path.join(data_path, 'HYDICE','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'HYDICE','groundtruth.mat'))['groundtruth']
    elif name == 'Airport':
        data = sio.loadmat(os.path.join(data_path, 'Airport','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Airport','groundtruth.mat'))['groundtruth']
    elif name == 'Airport2':
        data = sio.loadmat(os.path.join(data_path, 'Airport2','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Airport2','groundtruth.mat'))['groundtruth']
    elif name == 'Beach':
        data = sio.loadmat(os.path.join(data_path, 'Beach','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Beach','groundtruth.mat'))['groundtruth']
    elif name == 'Beach2':
        data = sio.loadmat(os.path.join(data_path, 'Beach2','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Beach2','groundtruth.mat'))['groundtruth']
    elif name == 'Urban1':
        data = sio.loadmat(os.path.join(data_path, 'Urban1','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Urban1','groundtruth.mat'))['groundtruth']
    elif name == 'Urban2':
        data = sio.loadmat(os.path.join(data_path, 'Urban2','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Urban2','groundtruth.mat'))['groundtruth']
    elif name == 'Segundo':
        data = sio.loadmat(os.path.join(data_path, 'Segundo','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Segundo','groundtruth.mat'))['groundtruth']
    elif name == 'Avon':
        data = sio.loadmat(os.path.join(data_path, 'Avon','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Avon','groundtruth.mat'))['groundtruth']
    elif name == 'Cuprite':
        data = sio.loadmat(os.path.join(data_path, 'Cuprite','agumentation_one.mat'))['agumentation_one']
        labels = sio.loadmat(os.path.join(data_path, 'Cuprite','groundtruth.mat'))['groundtruth']
    return data, labels

def loadData2(name):
    data_path = os.path.join(os.getcwd(), 'detectiondatasets')
    if name == 'Sandiego2':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego2', 'agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego2','groundtruth.mat'))['gt']
    elif name == 'Sandiego100':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'groundtruth.mat'))['groundtruth']
    elif name == 'MUUFL':
        data = sio.loadmat(os.path.join(data_path, 'MUUFL', 'agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'MUUFL', 'groundtruth.mat'))['groundtruth']
    elif name == 'GF5':
        data = sio.loadmat(os.path.join(data_path, 'GF5', 'agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'GF5', 'groundtruth.mat'))['groundtruth']
    elif name == 'Sandiego':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego', 'agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego', 'groundtruth.mat'))['groundtruth']
    elif name == 'Synthetic':
        data = sio.loadmat(os.path.join(data_path, 'Synthetic', 'agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Synthetic', 'groundtruth.mat'))['groundtruth']
    elif name == 'HYDICE':
        data = sio.loadmat(os.path.join(data_path, 'HYDICE','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'HYDICE','groundtruth.mat'))['groundtruth']
    elif name == 'Airport':
        data = sio.loadmat(os.path.join(data_path, 'Airport','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Airport','groundtruth.mat'))['groundtruth']
    elif name == 'Airport2':
        data = sio.loadmat(os.path.join(data_path, 'Airport2','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Airport2','groundtruth.mat'))['groundtruth']
    elif name == 'Beach':
        data = sio.loadmat(os.path.join(data_path, 'Beach','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Beach','groundtruth.mat'))['groundtruth']
    elif name == 'Beach2':
        data = sio.loadmat(os.path.join(data_path, 'Beach2','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Beach2','groundtruth.mat'))['groundtruth']
    elif name == 'Urban1':
        data = sio.loadmat(os.path.join(data_path, 'Urban1','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Urban1','groundtruth.mat'))['groundtruth']
    elif name == 'Urban2':
        data = sio.loadmat(os.path.join(data_path, 'Urban2','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Urban2','groundtruth.mat'))['groundtruth']
    elif name == 'Segundo':
        data = sio.loadmat(os.path.join(data_path, 'Segundo','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Segundo','groundtruth.mat'))['groundtruth']
    elif name == 'Avon':
        data = sio.loadmat(os.path.join(data_path, 'Avon','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Avon','groundtruth.mat'))['groundtruth']
    elif name == 'Cuprite':
        data = sio.loadmat(os.path.join(data_path, 'Cuprite','agumentation_two.mat'))['agumentation_two']
        labels = sio.loadmat(os.path.join(data_path, 'Cuprite','groundtruth.mat'))['groundtruth']
    return data, labels


