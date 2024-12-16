import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y, pred, title):
    # ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    fpr, tpr, threshold = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    print('roc_auc:', roc_auc)
    plt.style.use('ggplot')

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    # plt.savefig('result/'+title+'.svg', dpi=600)
    plt.savefig('detectiondatasets/'+title+ '/'+ title+ 'ROC' +'.jpg', bbox_inches= 'tight', pad_inches = 0, dpi = 1000)
    plt.show()

# Sandiego2
# data_name = 'Sandiego2'
# data_path = './detectiondatasets/Sandiego2/result.mat'
# gt_path = './detectiondatasets/Sandiego2/groundtruth.mat'

# MUUFL
data_name = 'MUUFL'
data_path = './detectiondatasets/MUUFL/result.mat'
gt_path = './detectiondatasets/MUUFL/groundtruth.mat'


# GF5
# data_name = 'GF5'
# data_path = './detectiondatasets/GF5/result.mat'
# gt_path = './detectiondatasets/GF5/groundtruth.mat'



# Sandiego100
# data_name = 'Sandiego100'
# data_path = './detectiondatasets/Sandiego100/result.mat'
# gt_path = './detectiondatasets/Sandiego100/groundtruth.mat'


# HYDICE
# data_name = 'HYDICE'
# data_path = './detectiondatasets/HYDICE/result.mat'
# gt_path = './detectiondatasets/HYDICE/groundtruth.mat'



# Urban1
# data_name = 'Urban1'
# data_path = './detectiondatasets/Urban1/result.mat'
# gt_path = './detectiondatasets/Urban1/groundtruth.mat'


# Airport
# data_name = 'Airport'
# data_path = './detectiondatasets/Airport/result.mat'
# gt_path = './detectiondatasets/Airport/groundtruth.mat'

# Synthetic
# data_name = 'Synthetic'
# data_path = './detectiondatasets/Synthetic/result.mat'
# gt_path = './detectiondatasets/Synthetic/groundtruth.mat'


# Segundo
# data_name = 'Segundo'
# data_path = './detectiondatasets/Segundo/result.mat'
# gt_path = './detectiondatasets/Segundo/groundtruth.mat'



data = sio.loadmat(data_path)['result']
plt.figure()
plt.imshow(data, cmap='afmhot')
plt.axis('off')
pathfigure = './detectiondatasets/' + data_name + '/' + data_name + '.jpg'
plt.savefig(pathfigure, bbox_inches='tight', pad_inches=0, dpi=1000)
plt.show()

gt = sio.loadmat(gt_path)['groundtruth']
# gt = sio.loadmat(gt_path)['gt']  # Sandiego2 dedicated
data = data.reshape(-1)
gt = gt.reshape(-1)

plot_roc_curve(gt, data, data_name)

