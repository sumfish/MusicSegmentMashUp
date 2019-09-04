import torch 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import dataset
import numpy as np
from torch.utils.data import DataLoader
from model import CDSSM

neg_num = 2 # Number of random unclicked documents serving as negative examples for a query. See section 4.
nums_epoch=50
data_path='./audio_docu/'
batch_size=16  #no16
test_batch_size=100

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
train_data=dataset.MusicSet('data_train.txt')
trainloader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
test_data=dataset.MusicSet('data_test.txt')
testloader = DataLoader(dataset=test_data, batch_size=test_batch_size,shuffle=True)

# model
#model = CDSSM().to(device)
model = CDSSM()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# output variable, remember the cosine similarity with positive doc was at 0th index

for epoch in range(nums_epoch):
    model.train()
    for batch_idx, (Q, P, N) in enumerate(trainloader):
        '''
        N_list=[]
        Q, P= Q.to(device), P.to(device)
        for i in range(J):
            tensor=N[i].to(device)
            N_list.append(tensor)
        model=model.double().to(device)
        y_pred = model(Q, P, N_list)
        '''
        model=model.double()
        y_pred = model(Q, P, N)
        #print(y_pred)
        n_correct=0
        for i in range(len(y_pred)):
            record=y_pred[i].resize(1,neg_num+1).float()
            _,idx=torch.max(record,dim=1)
            #print(record)
            #print(idx)
            if(idx==0):
                n_correct+=1
            if i==0:
                a=record
            else:
                b=record
                a=torch.cat((a,b),0)
            #y_pred_list.append(y_pred[i].resize(1,J+1))
        acc=float(n_correct)/len(y_pred)
        #print(acc)
        y = np.ndarray(len(y_pred))
        # CrossEntropyLoss expects only the index as a long tensor
        y[:] = 0
        y = Variable(torch.from_numpy(y).long())
        loss = criterion(a, y) #Input&target
        #print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            '''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            '''
            print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.6f}, Acc: {:.6f}" 
                   .format(epoch+1, nums_epoch, batch_idx+1, len(trainloader), loss.item(),acc))
