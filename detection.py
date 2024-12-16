import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

from models import SelfSupConNet
from utils import plot_roc_curve
from sklearn.metrics.pairwise import laplacian_kernel, sigmoid_kernel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sandiego2 189bands AVIRIS
# data_name = 'Sandiego2'
# data_name = 'Sandiego100'
data_name = 'MUUFL'
# data_name = 'GF5'
# data_name = 'Sandiego'
# data_name = 'Synthetic'
# data_name = 'Avon'
# data_name = 'Cuprite'
# data_name = 'HYDICE'
# data_name = 'Airport'
# data_name = 'Airport2'
# data_name = 'Beach'
# data_name = 'Beach2'
# data_name = 'Urban1'
# data_name = 'Urban2'
# data_name = 'Segundo'




# hyperdata_path1 = './detectiondatasets/' + data_name + '/agumentation_one.mat'
# hyperdata_path2 = './detectiondatasets/' + data_name + '/agumentation_two.mat'
# gdt_path = './detectiondatasets/'+ data_name + '/groundtruth.mat'
# prior_path1 = './detectiondatasets/'+ data_name+'/target_one.mat'
# prior_path2 = './detectiondatasets/'+ data_name+'/target_two.mat'
#
# hyperdata1 = sio.loadmat(hyperdata_path1)['agumentation_one']
# hyperdata2 = sio.loadmat(hyperdata_path2)['agumentation_two']
# hyperdata1 = hyperdata1.reshape(100*100,94)
# hyperdata2 = hyperdata2.reshape(100*100,94)
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior1 = sio.loadmat(prior_path1)['target_one']
# prior2 = sio.loadmat(prior_path2)['target_two']

# Sandiego2
# hyperdata_path = './detectiondatasets/' + data_name + '/sandiego.mat'
# gdt_path = './detectiondatasets/' + data_name + './groundtruth.mat'
# prior_path = './detectiondatasets/' + data_name + './prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(120*120,-1)
# gdt = sio.loadmat(gdt_path)['gt']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0

# Sandiego100
# hyperdata_path = './detectiondatasets/' + data_name + '/sandiego.mat'
# gdt_path = './detectiondatasets/' + data_name + './groundtruth.mat'
# prior_path = './detectiondatasets/' + data_name + './prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(100*100,-1)
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0


# Urban1
# hyperdata_path = './detectiondatasets/' + data_name + '/data.mat'
# gdt_path = './detectiondatasets/' + data_name + './groundtruth.mat'
# prior_path = './detectiondatasets/' + data_name + './prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(100*100,-1)
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0


# MUUFL
hyperdata_path = './detectiondatasets/' + data_name + '/data.mat'
gdt_path = './detectiondatasets/' + data_name + './groundtruth.mat'
prior_path = './detectiondatasets/' + data_name + './prior_target.mat'
hyperdata = sio.loadmat(hyperdata_path)['data']
hyperdata = hyperdata.reshape(325*220,-1)
gdt = sio.loadmat(gdt_path)['groundtruth']
gdt = gdt.reshape(-1)
prior = sio.loadmat(prior_path)['prior_target'].T
prior = prior/1.0

# GF5
# hyperdata_path = './detectiondatasets/' + data_name + '/data.mat'
# gdt_path = './detectiondatasets/' + data_name + './groundtruth.mat'
# prior_path = './detectiondatasets/' + data_name + './prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(100*250,-1)
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0



# Synthetic
# hyperdata_path = './detectiondatasets/' + data_name + '/synthetic.mat'
# gdt_path = './detectiondatasets/' + data_name + './groundtruth.mat'
# prior_path = './detectiondatasets/' + data_name + './prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(64*64,-1)
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0


# Avon
# hyperdata_path = './detectiondatasets/' + data_name + '/avon.mat'
# gdt_path = './detectiondatasets/' + data_name + './groundtruth.mat'
# prior_path = './detectiondatasets/' + data_name + './prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(100*100,-1)
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0


# Cuprite
# hyperdata_path = './detectiondatasets/' + data_name + '/cuprite.mat'
# gdt_path = './detectiondatasets/' + data_name + './groundtruth.mat'
# prior_path = './detectiondatasets/' + data_name + './prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(100*100,-1)
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0



# hyperdata_path = './detectiondatasets/' + data_name + '/AAE_Features2.mat'
# gdt_path = './detectiondatasets/' + data_name + './y1.mat'
# prior_path = './detectiondatasets/' + data_name + './target2.mat'
# hyperdata = sio.loadmat(hyperdata_path)['AAE_Features2']
#
# gdt = sio.loadmat(gdt_path)['y1']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['tgt']

def cos_sim(vector_1, vector_2):
    # return -20 * math.log(
    #     math.acos(np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2)))), 2)
    # return 99999999999999990000000000000000000000000000000000000000000000000000000000000000000000000000000000000000**(np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2))))
    # return 9000000000000000000000000000000000000000 ** (
    #             np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2))))
    return np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2)))
    # return 90000000000000000000000000 ** (
    #         np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2))))
    # return 10 ** (
    #         np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2))))



class getDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data/1.0
        self.labels = label
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        index = index
        Data = (self.data[index])
        label = (self.labels[index])
        return torch.FloatTensor(Data), label

detect_dataset = getDataset(hyperdata, gdt)
detect_loader = torch.utils.data.DataLoader(detect_dataset, batch_size=1)

model = SelfSupConNet()

#Sandiego2
# ckpt = torch.load('./save/SelfCon/Sandiego2_models/SimCLR_Sandiego2_ConResNet_lr_0.05_decay_0.0001_bsz_240_temp_0.1_trial_10/last.pth', map_location='cpu')

#Sandiego100
# ckpt = torch.load('./save/SelfCon/Sandiego100_models/SimCLR_Sandiego100_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_10/last.pth', map_location='cpu')

# MUUFL
ckpt = torch.load('./save/SelfCon/MUUFL_models/SimCLR_MUUFL_ConResNet_lr_0.05_decay_0.0001_bsz_500_temp_0.1_trial_no_augmentation_warm/last.pth', map_location='cpu')
# ckpt = torch.load('./save/SelfCon/MUUFL_models/SimCLR_MUUFL_ConResNet_lr_0.05_decay_0.0001_bsz_500_temp_0.1_trial_11_warm/last.pth', map_location='cpu')



# GF5
# ckpt = torch.load('./save/SelfCon/GF5_models/SimCLR_GF5_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_1/last.pth', map_location='cpu')



# Sandiego
# ckpt = torch.load('./save/SelfCon/Sandiego_models/SimCLR_Sandiego_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_0/last.pth', map_location='cpu')

# Synthetic
# ckpt = torch.load('./save/SelfCon/Synthetic_models/SimCLR_Synthetic_ConResNet_lr_0.05_decay_0.0001_bsz_128_temp_0.1_trial_0/last.pth', map_location='cpu')


# Avon
# ckpt = torch.load('./save/SelfCon/Avon_models/SimCLR_Avon_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_0/last.pth', map_location='cpu')


# Cuprite
# ckpt = torch.load('./save/SelfCon/Cuprite_models/SimCLR_Cuprite_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_0/last.pth', map_location='cpu')


# HYDICE
# ckpt = torch.load('./save/SelfCon/HYDICE_models/SimCLR_HYDICE_ConResNet_lr_0.05_decay_0.0001_bsz_160_temp_0.07_trial_0/last.pth', map_location='cpu')


#Airport
# ckpt = torch.load('./save/SelfCon/Airport_models/SimCLR_Airport_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.05_trial_1/last.pth', map_location='cpu')

# Airport2
# ckpt = torch.load('./save/SelfCon/Airport2_models/SimCLR_Airport2_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_1/last.pth', map_location='cpu')

# Beach
# ckpt = torch.load('./save/SelfCon/Beach_models/SimCLR_Beach_ConResNet_lr_0.05_decay_0.0001_bsz_300_temp_0.1_trial_0_warm/last.pth', map_location='cpu')

# Beach2
# ckpt = torch.load('./save/SelfCon/Beach2_models/SimCLR_Beach2_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_0/last.pth', map_location='cpu')

# Urban1
# ckpt = torch.load('./save/SelfCon/Urban1_models/SimCLR_Urban1_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_11/last.pth', map_location='cpu')

# Urban2
# ckpt = torch.load('./save/SelfCon/Urban2_models/SimCLR_Urban2_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_0/last.pth', map_location='cpu')

# Segundo
# ckpt = torch.load('./save/SelfCon/Segundo_models/SimCLR_Segundo_ConResNet_lr_0.05_decay_0.0001_bsz_200_temp_0.1_trial_1/last.pth', map_location='cpu')



# ckpt = torch.load('./save/SelfCon/Salinas_models/SimCLR_Salinas_ConResNet_lr_0.05_decay_0.0001_bsz_128_temp_0.09_trial_0/last.pth', map_location='cpu')


state_dict = ckpt['model']
if torch.cuda.is_available():
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
model = model.cuda()
model.load_state_dict(state_dict)

# prior1 = torch.FloatTensor(prior1)
# prior2 = torch.FloatTensor(prior2)

start = time.time()
model.eval()
target_detector = []
feature = []

# length, bands = prior.shape[0], prior.shape[1]
# feature = np.zeros(10000)
prior = torch.FloatTensor(prior)
# with torch.no_grad():
    # for i in range(length):
    #     prior1 = prior[i]
    #
    #     prior1 = torch.unsqueeze(prior1, 0)
    #     prior1 = prior1.to(device)
    #     prior_output = model.encoder(prior1)
    #     prior_output = prior_output.cuda().data.cpu().numpy()
    #     for idx, (images, labels) in enumerate(detect_loader):
    #         # images = torch.unsqueeze(images, 0)
    #         images = images.to(device)
    #         outputs = model.encoder(images)
    #         outputs = outputs.cuda().data.cpu().numpy()
    #         # detection = pairwise_distances(prior_output, outputs, metric='manhattan')
    #         detection = cos_sim(prior_output, outputs)
    #         detection = detection.squeeze()
    #         target_detector.append(detection)
    #     feature = np.array(feature) + np.array(target_detector)
    #     target_detector = []


with torch.no_grad():
    # prior = torch.unsqueeze(prior, 0)
    prior = prior.to(device)
    prior_output = model.encoder(prior)
    prior_output = prior_output.cuda().data.cpu().numpy()


    # for images, _ in detect_loader:
    for idx, (images, labels) in enumerate(detect_loader):
        # images = torch.unsqueeze(images, 0)
        a = images.size(0)
        # prior_output = np.tile(prior_output,(a,1))
        images = images.to(device)
        outputs = model.encoder(images)
        outputs = outputs.cuda().data.cpu().numpy()
        # detection = pairwise_distances(prior_output, outputs, metric='manhattan')
        detection = cos_sim(prior_output, outputs)
        # detection = detection[1,:]#.squeeze()
        # detection = detection.tolist()
        target_detector.extend(detection)
        # feature.append(outputs)

end = time.time()
print('running time:', end - start)

target_detector = np.array(target_detector)
# target_detector = target_detector/length
# feature = np.squeeze(feature)
# feature = np.array(feature)

max3 = np.amax(target_detector)
min3 = np.amin(target_detector)
target_detector = (target_detector - min3)/(max3 - min3)


plot_roc_curve(gdt, target_detector, data_name)

# Sandiego2
# target_detector = np.reshape(target_detector, (120,120))

# Sandiego100
# target_detector = np.reshape(target_detector, (100,100))

# MUUFL
target_detector = np.reshape(target_detector, (325,220))

# GF5
# target_detector = np.reshape(target_detector, (100,250))



# Sandiego
# target_detector = np.reshape(target_detector, (100,100))


# Synthetic
# target_detector = np.reshape(target_detector, (64,64))

# Avon
# target_detector = np.reshape(target_detector, (100,100))

# Cuprite
# target_detector = np.reshape(target_detector, (100,100))



# HYDICE
# target_detector = np.reshape(target_detector, (80,100))


# Airport
# target_detector = np.reshape(target_detector, (100,100))

# Airport2
# target_detector = np.reshape(target_detector, (100,100))

# Beach
# target_detector = np.reshape(target_detector, (150,150))

# Beach2
# target_detector = np.reshape(target_detector, (100,100))

# Urban1
# target_detector = np.reshape(target_detector, (100,100))

# Urban2
# target_detector = np.reshape(target_detector, (100,100))

# Segundo
# target_detector = np.reshape(target_detector, (100,100))




target_detector = target_detector.tolist()
plt.figure(2)
plt.imshow(target_detector, cmap='afmhot')
# plt.imshow(target_detector)
plt.axis('off')
pathfigure = './result/' + data_name + '.jpg'
plt.savefig(pathfigure, bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()


path_target_detector = './result/' + data_name + '.mat'
sio.savemat(path_target_detector, {'detect': target_detector})

# path_feature = './detectiondatasets/' + data_name +'/feature.mat'
# sio.savemat(path_feature, {'feature': feature})
