import numpy as np
from scipy.io import loadmat, savemat
from spectral import *
import matplotlib.pyplot as plt

# input_data = loadmat('./detectiondatasets/Sandiego2/sandiego.mat')['data']
# input_data = loadmat('./detectiondatasets/Sandiego100/sandiego.mat')['data']
# input_data = loadmat('./detectiondatasets/Airport/airport.mat')['data']
# input_data = loadmat('./detectiondatasets/Synthetic/synthetic.mat')['data']
# input_data = loadmat('./detectiondatasets/MUUFL/data.mat')['data']
input_data = loadmat('./detectiondatasets/GF5/data.mat')['data']


# input_data = loadmat('./detectiondatasets/HYDICE/hydice_urban.mat')['data']
# input_data = loadmat('./detectiondatasets/Segundo/segundo.mat')['data']


# input_data = loadmat('./detectiondatasets/Urban1/data.mat')['data']

input_data = np.float32(input_data)
max1 = np.amax(input_data)
min1 = np.amin(input_data)
input_data = (input_data-min1)/(max1-min1)
pc = principal_components(input_data)
pc_099 = pc.reduce(fraction=0.99)
firstpca = pc_099.transform(input_data)
# plt.imshow(firstpca[:, :, 0], cmap='afmhot')     # Segundo=0, Sandiego100=1, Urban=0, Sandiego2=0, Beach=0, HYDICE=2
# plt.axis('off')
# plt.show()

# path = './detectiondatasets/Sandiego2/firstpca.mat'
# path = './detectiondatasets/Sandiego100/firstpca.mat'
# path = './detectiondatasets/Airport/firstpca.mat'
# path = './detectiondatasets/Synthetic/firstpca.mat'
# path = './detectiondatasets/MUUFL/firstpca.mat'
path = './detectiondatasets/GF5/firstpca.mat'


# path = './detectiondatasets/HYDICE/firstpca.mat'
# path = './detectiondatasets/Urban1/firstpca.mat'
# path = './detectiondatasets/Segundo/firstpca.mat'


savemat(path,{'firstpca': firstpca})

