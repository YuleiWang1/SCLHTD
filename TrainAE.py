import numpy as np
import torch

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) -1)

# class AEDataset(torch.utils.data.Dataset):
#     def __init__(self, Datapath, transform):
#         self.Datalist = np.load(Datapath)
#         self.transform = transform
#     def __getitem__(self, index):
#         Data = self.transform(self.Datalist[index].astype('float64'))
#         Data = Data.view(1, Data.shape[0], Data.shape[1], Data.shape[2])
#         return Data
#     def __len__(self):
#         return len(self.Datalist)

class AEDataset(torch.utils.data.Dataset):
    def __init__(self, Datapath):
        self.Datalist = np.load(Datapath)
    def __getitem__(self, index):
        Data = self.Datalist[index].astype('float64')
        Data = torch.FloatTensor(Data)
        Data = torch.unsqueeze(Data, 0)
        return Data
    def __len__(self):
        return len(self.Datalist)



def generate_(batch_size, dim):
    return torch.from_numpy(
        np.random.multivariate_normal(mean=np.zeros([dim]), cov=np.diag(np.ones([dim])), size=batch_size)
    ).type(torch.float)

def TrainAAE_patch1(XPath, Patch_channel, windowSize, encoded_dim, batch_size):
    import torch
    from AugmentationModels import Enc_AAE, Dec_AAE, Discriminant
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_AAE(channel=Patch_channel,output_dim=encoded_dim,windowSize=windowSize).cuda()
    Dec_patch = Dec_AAE(channel=Patch_channel,windowSize=windowSize,input_dim=encoded_dim).cuda()
    discriminant = Discriminant(encoded_dim).cuda()

    ##训练patchAE
    # patch_data = AEDataset(XPath,trans)
    patch_data = AEDataset(XPath)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=True)
    optim_enc = torch.optim.Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_dec=torch.optim.Adam(Dec_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_enc_gen = torch.optim.SGD(Enc_patch.parameters(), lr=1e-4, weight_decay=0.000)  # 1e-5
    optim_disc = torch.optim.SGD(discriminant.parameters(), lr=5e-5, weight_decay=0.000)  # 5e-6
    criterion = torch.nn.MSELoss() 
    epochs=20
    for epoch in range(epochs):
        rl=0
        l_dis_loss=0
        l_encl=0
        print('Epoch No {}'.format(epoch))
        for i, (data) in enumerate(tqdm(Patch_loader)):
            ######################### reconstruction phase
            data=data.cuda().float()
            Enc_patch.train()
            Dec_patch.train()
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            optim_disc.zero_grad()
            optim_enc_gen.zero_grad()
            map, code =Enc_patch(data)
            recon=Dec_patch(code)
            loss=criterion(data,recon)
            loss.backward(retain_graph=True)
            ####################################### regularization phase
            discriminant.train()
            Enc_patch.eval()
            gauss=torch.FloatTensor(generate_(batch_size,encoded_dim)).cuda()
            fake_pred = discriminant(gauss)
            true_pred = discriminant(code)
            dis_loss=-(torch.mean(fake_pred) -torch.mean(true_pred))
            dis_loss.backward(retain_graph=True)
            discriminant.train()
            Enc_patch.train()
            encl=-torch.mean(true_pred)
            encl.backward(retain_graph=True)
            optim_dec.step()
            optim_enc.step()
            optim_disc.step()
            optim_enc_gen.step()
            rl = rl + loss.item()
            l_dis_loss+=dis_loss.item()
            l_encl+=encl.item()
        print('\nPatch Reconstruction Loss: {}  dis loss: {}   regularization loss : {}'.format(rl/len(patch_data),l_dis_loss/len(patch_data),l_encl/len(patch_data)))
    torch.save(Enc_patch.state_dict(),'./models/Enc_AAE1.pth')
    return 0

def TrainAAE_patch2(XPath, Patch_channel, windowSize, encoded_dim, batch_size):
    import torch
    from AugmentationModels import Enc_AAE, Dec_AAE, Discriminant
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_AAE(channel=Patch_channel,output_dim=encoded_dim,windowSize=windowSize).cuda()
    Dec_patch = Dec_AAE(channel=Patch_channel,windowSize=windowSize,input_dim=encoded_dim).cuda()
    discriminant = Discriminant(encoded_dim).cuda()

    ##训练patchAE
    # patch_data = AEDataset(XPath,trans)
    patch_data = AEDataset(XPath)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=True)
    optim_enc = torch.optim.Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_dec=torch.optim.Adam(Dec_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_enc_gen = torch.optim.SGD(Enc_patch.parameters(), lr=1e-4, weight_decay=0.000)  # 1e-5
    optim_disc = torch.optim.SGD(discriminant.parameters(), lr=5e-5, weight_decay=0.000)  # 5e-6
    criterion = torch.nn.MSELoss() 
    epochs=20
    for epoch in range(epochs):
        rl=0
        l_dis_loss=0
        l_encl=0
        print('Epoch No {}'.format(epoch))
        for i, (data) in enumerate(tqdm(Patch_loader)):
            ######################### reconstruction phase
            data=data.cuda().float()
            Enc_patch.train()
            Dec_patch.train()
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            optim_disc.zero_grad()
            optim_enc_gen.zero_grad()
            map, code =Enc_patch(data)
            recon=Dec_patch(code)
            loss=criterion(data,recon)
            loss.backward(retain_graph=True)
            ####################################### regularization phase
            discriminant.train()
            Enc_patch.eval()
            gauss=torch.FloatTensor(generate_(batch_size,encoded_dim)).cuda()
            fake_pred = discriminant(gauss)
            true_pred = discriminant(code)
            dis_loss=-(torch.mean(fake_pred) -torch.mean(true_pred))
            dis_loss.backward(retain_graph=True)
            discriminant.train()
            Enc_patch.train()
            encl=-torch.mean(true_pred)
            encl.backward(retain_graph=True)
            optim_dec.step()
            optim_enc.step()
            optim_disc.step()
            optim_enc_gen.step()
            rl = rl + loss.item()
            l_dis_loss+=dis_loss.item()
            l_encl+=encl.item()
        print('\nPatch Reconstruction Loss: {}  dis loss: {}   regularization loss : {}'.format(rl/len(patch_data),l_dis_loss/len(patch_data),l_encl/len(patch_data)))
    torch.save(Enc_patch.state_dict(),'./models/Enc_AAE2.pth')
    return 0

def SaveFeatures_AAE1(XPath,Patch_channel=15,windowSize=25,encoded_dim=64,batch_size=128):
    import torch
    from AugmentationModels import  Enc_AAE
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    from Preprocess import feature_normalize2, L2_Norm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_AAE(channel=Patch_channel,output_dim=encoded_dim,windowSize=windowSize).cuda()
    Enc_patch.load_state_dict(torch.load('./models/Enc_AAE1.pth'))
    ##运行patchAE 的encoder
    patch_data = AEDataset(XPath)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=False)
    Patch_Features=[]
    print('Start save patch features...')
    for i, (data) in enumerate(tqdm(Patch_loader)):
        data=data.cuda().float()
        Enc_patch.eval()
        feature,code= Enc_patch(data)
        for num in range(len(feature)):
            Patch_Features.append(np.array(feature[num].cpu().detach().numpy()))
    # Patch_Features=feature_normalize2(Patch_Features)
    Patch_Features = L2_Norm(Patch_Features)

    return Patch_Features

def SaveFeatures_AAE2(XPath,Patch_channel=15,windowSize=25,encoded_dim=64,batch_size=128):
    import torch
    from AugmentationModels import  Enc_AAE
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    from Preprocess import feature_normalize2, L2_Norm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_AAE(channel=Patch_channel,output_dim=encoded_dim,windowSize=windowSize).cuda()
    Enc_patch.load_state_dict(torch.load('./models/Enc_AAE2.pth'))
    ##运行patchAE 的encoder
    patch_data = AEDataset(XPath)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=False)
    Patch_Features=[]
    print('Start save patch features...')
    for i, (data) in enumerate(tqdm(Patch_loader)):
        data=data.cuda().float()
        Enc_patch.eval()
        feature,code= Enc_patch(data)
        for num in range(len(feature)):
            Patch_Features.append(np.array(feature[num].cpu().detach().numpy()))
    # Patch_Features=feature_normalize2(Patch_Features)
    Patch_Features = L2_Norm(Patch_Features)

    return Patch_Features


