import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision

minibatch_size=32

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=minibatch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
batch_size=minibatch_size, shuffle=True, **kwargs)

def num_flat_features(x):
    size=x.size()[1:]
    num_features = 1
    for s in size:
        num_features *=s
    return num_features

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, 1024)
        self.map2 = nn.Linear(1024, 7*7*128)
        self.bm1 = nn.BatchNorm1d(1024)
        self.bm2 = nn.BatchNorm1d(7*7*128)
        self.bm3 = nn.BatchNorm2d(64)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)
    def forward(self, x):
        x = F.leaky_relu(self.bm1(self.map1(x)))
        x = F.leaky_relu(self.bm2(self.map2(x)))

        x = x.view(-1,128,7,7)

        x = self.upconv1(x)
        x = self.bm3(x)
        x = F.leaky_relu(x)

        return self.upconv2(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1,64,4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64,128,4,stride=2,padding=1)
        self.map1 = nn.Linear(128*7*7, 1024)
        self.map2 = nn.Linear(1024, 1)
        self.bm1 = nn.BatchNorm2d(64)
        self.bm2 = nn.BatchNorm2d(128)
        self.bm3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bm2(x)
        x = F.leaky_relu(x)

        x = x.view(-1,num_flat_features(x))
        x = self.map1(x)
        x = self.bm3(x)
        x = F.leaky_relu(x)

        return self.map2(x)

class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1,64,4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64,128,4,stride=2,padding=1)
        self.map1 = nn.Linear(128*7*7, 1024)
        self.map2 = nn.Linear(1024, input_size)
        self.bm1=nn.BatchNorm2d(64)
        self.bm2=nn.BatchNorm2d(128)
        self.bm3=nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bm2(x)
        x = F.leaky_relu(x)

        x = x.view(-1,num_flat_features(x))
        x = self.map1(x)
        x = self.bm3(x)
        x = F.leaky_relu(x)

        return self.map2(x)

class Codiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Codiscriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

        self.bm1 = nn.BatchNorm1d(input_size)
        self.bm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.leaky_relu(self.bm1(self.map1(x)))
        x = F.leaky_relu(self.bm2(self.map2(x)))
        return self.map3(x)

z_dim=50
lambdy=1
g_learning_rate=0.001
d_learning_rate=0.0001
optim_betas = (0.5, 0.9)
num_epochs=40
epsil = 1e-8

G = Generator(input_size=z_dim).cuda()
E = Encoder(z_dim).cuda()
D = Discriminator().cuda()
C = Codiscriminator(input_size=z_dim, hidden_size=z_dim, output_size=1).cuda()


criterion_l1 = nn.L1Loss()
def criterion(a, b) :
    crit=nn.BCELoss()
    ash=crit(torch.clamp(a, min=epsil, max=1-epsil), b)
    return ash

d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
e_optimizer = optim.Adam(E.parameters(), lr=g_learning_rate, betas=optim_betas)
c_optimizer = optim.Adam(C.parameters(), lr=d_learning_rate, betas=optim_betas)


for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
            E.zero_grad()
            G.zero_grad()
            C.zero_grad()
            D.zero_grad()

            x_real = Variable(data).cuda()
            z_real = Variable(torch.randn(minibatch_size, z_dim)).cuda()
            x_fake = G(z_real)
            z_fake = E(x_real)
            x_reco = G(z_fake)

            d_real = D(x_real)
            d_fake = D(x_fake)
            d_reco = D(x_reco)

            c_real = C(z_real)
            c_fake = C(z_fake)

            g_real_error = criterion(d_fake, Variable(torch.ones(minibatch_size).cuda()))  # we want to fool, so pretend it's all genuine
            g_rec_error = lambdy*criterion_l1(x_reco,x_real)+criterion(d_reco, Variable(torch.ones(minibatch_size).cuda()))
            e_error = criterion(c_fake, Variable(torch.ones(minibatch_size).cuda()))
            ge_error=g_real_error+g_rec_error+e_error

            d_real_error = criterion(d_real, Variable(torch.ones(minibatch_size).cuda()))
            d_fake_error = criterion(d_fake, Variable(torch.zeros(minibatch_size).cuda()))
            d_rec_error = criterion(d_reco, Variable(torch.zeros(minibatch_size).cuda()))

            c_real_error = criterion(c_real, Variable(torch.ones(minibatch_size).cuda()))
            c_fake_error = criterion(c_fake, Variable(torch.zeros(minibatch_size).cuda()))
            c_error = c_real_error+c_fake_error

            ge_error.backward(retain_variables=True)
            g_optimizer.step()
            e_optimizer.step()

            C.zero_grad()
            c_error.backward(retain_variables=True)
            c_optimizer.step()

            D.zero_grad()
            d_error.backward(retain_variables=True)
            d_optimizer.step()
    print "%d epoch" %epoch
    print "ge_error : %s    c_error : %s    d_error : %s" %(ge_error.data, c_error.data, d_error.data)
    tasty=Variable(torch.randn(100,z_dim)).cuda()
    tense=G(tasty).view(-1,1,28,28).cpu()
    tt=torch.clamp(tense,0,1)
    torchvision.utils.save_image(tt.data,'resulty'+str(epoch)+'.png')
