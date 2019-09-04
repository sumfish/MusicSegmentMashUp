import torch 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import dataset
import numpy as np
from torch.utils.data import DataLoader
from visdom import Visdom
from model import CDSSM

neg_num = 2 # Number of random unclicked documents serving as negative examples for a query. See section 4.
n_epochs=100
data_path='./audio_docu/'
batch_size=16  #no16
test_batch_size=100
log_interval=10

def main():
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

    # draw
    '''
    vis = Visdom(env='music')
    vis.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    '''
    for epoch in range(n_epochs):
        # train stage
        train_loss = train(device, model, trainloader, criterion, optimizer)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}\n'.format(epoch + 1, n_epochs, train_loss)
        #vis.line([train_loss], [epoch+1], win='train_loss', update='append')

        # test stage
        val_loss = test(device, model, testloader, criterion, optimizer)
        #val_loss /= len(val_loader)
        message += 'Epoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
        print(message)

def train(device, model, trainloader, criterion, optimizer):
    model.train()
    total_loss=0
    losses=[]

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

        ### CrossEntropyLoss expects only the index as a long tensor
        y = np.ndarray(len(y_pred))
        y[:] = 0
        y = Variable(torch.from_numpy(y).long())
        loss = criterion(a, y) #Input&target
        #print(loss)

        losses.append(loss.item())
        total_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            #average loss 
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: {:.6f}'.format(
                batch_idx * len(Q), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), np.mean(losses),acc)
            print(message)
            # print(len(trainloader))
            loss=[]

    total_loss/=(batch_idx+1)
    return total_loss

def test(device, model, testloader, criterion, optimizer):
    with torch.no_grad():
        model.eval()
        val_loss=0
        losses=[]
        for batch_idx, (Q, P, N) in enumerate(testloader):
        
            model=model.double()
            y_pred = model(Q, P, N)
            
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

            ### CrossEntropyLoss expects only the index as a long tensor
            y = np.ndarray(len(y_pred))
            y[:] = 0
            y = Variable(torch.from_numpy(y).long())
            loss = criterion(a, y) #Input&target
            #print(loss)

            losses.append(loss.item())
            val_loss+=loss.item()

            if batch_idx % log_interval == 0:
                #average loss 
                message = 'Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: {:.6f}'.format(
                    batch_idx * len(Q), len(testloader.dataset),
                    100. * batch_idx / len(testloader), np.mean(losses),acc)
                print(message)
                loss=[]

    return val_loss


if __name__=='__main__':
    main()