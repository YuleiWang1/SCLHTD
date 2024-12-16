import argparse
import time
from Preprocess import Preprocess1, Preprocess2
from TrainAE import TrainAAE_patch1, TrainAAE_patch2, SaveFeatures_AAE1, SaveFeatures_AAE2
import numpy as np

dataset_names = ['Sandiego2', 'Sandiego100', 'HYDICE', 'Airport', 'Airport2', 'Beach', 'Beach2', 'Urban1', 'Urban2', 'Segundo', 'Sandiego', 'Synthetic', 'Avon', 'Cuprite', 'MUUFL', 'GF5']
parser = argparse.ArgumentParser(description="Data Agumentation")
parser.add_argument('--dataset', type=str, default='MUUFL')
parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0","cuda:1"))
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--encoded_dim', type=int, default=64)
parser.add_argument('--Windowsize', type=int, default=3)
parser.add_argument('--Patch_channel', type=int, default=32)   # GF5=75; MUUFL=32; Sandiego100=94; Sandiego2=94; HYDICE=87; Airport=95; Airport2=102; Beach=51; Urban1=102; Urban2=103; Beach2=94; Segundo=112; Sandiego=63; Synthetic=112; Avonn=180; Cuprite=94
parser.add_argument('--train',type=int, default=1,choices=(0,1))

args = parser.parse_args()
print(args)

# Datadir = './detectiondatasets/Sandiego2/'
# Datadir = './detectiondatasets/Sandiego100/'
Datadir = './detectiondatasets/MUUFL/'
# Datadir = './detectiondatasets/GF5/'

# Datadir = './detectiondatasets/Sandiego/'
# Datadir = './detectiondatasets/Synthetic/'
# Datadir = './detectiondatasets/Avon/'
# Datadir = './detectiondatasets/Cuprite/'


# Datadir = './detectiondatasets/HYDICE/'
# Datadir = './detectiondatasets/Airport/'
# Datadir = './detectiondatasets/Airport2/'
# Datadir = './detectiondatasets/Beach/'
# Datadir = './detectiondatasets/Beach2/'
# Datadir = './detectiondatasets/Urban1/'
# Datadir = './detectiondatasets/Urban2/'
# Datadir = './detectiondatasets/Segundo/'



XPath1 = Datadir + 'X1.npy'
yPath1 = Datadir + 'y1.npy'
XPath2 = Datadir + 'X2.npy'
yPath2 = Datadir + 'y2.npy'
start = time.time()
if args.train:
    Preprocess1(XPath1, yPath1, args.dataset, args.Windowsize, Patch_channel=args.Patch_channel)
    TrainAAE_patch1(XPath1, Patch_channel=args. Patch_channel, windowSize=args.Windowsize, encoded_dim=args.encoded_dim, batch_size=args.batch_size)
    Preprocess2(XPath2, yPath2, args.dataset, args.Windowsize, Patch_channel=args.Patch_channel)
    TrainAAE_patch2(XPath2, Patch_channel=args. Patch_channel, windowSize=args.Windowsize, encoded_dim=args.encoded_dim, batch_size=args.batch_size)

    
AAEPath1=Datadir+'AAE_Features1.npy'
AAEPath2=Datadir+'AAE_Features2.npy'

AAEFeatures1=SaveFeatures_AAE1(XPath1,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)

AAEFeatures2=SaveFeatures_AAE2(XPath2,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)

end = time.time()
print('data agumentation time:', end - start)

np.save(AAEPath1,AAEFeatures1)
np.save(AAEPath2,AAEFeatures2)