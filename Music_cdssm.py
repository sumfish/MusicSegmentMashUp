import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable

LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
TOTAL_LETTER_GRAMS = int(3 * 1e4) # Determined from data. See section 3.2.
WORD_DEPTH = 128 # See equation (1).
# Uncomment it, if testing
# WORD_DEPTH = 1000
K = 64 # Dimensionality of the max-pooling layer. See section 3.4.
L = 32 # Dimensionality of latent semantic space. See section 3.5.
J = 2 # Number of random unclicked documents serving as negative examples for a query. See section 4.
# kernel size of time(word_depth*time)
FILTER_LENGTH = 1 # We only consider one time step for convolutions.
nums_epoch=50
data_path='./audio_docu/'
batch_size=8  #no16
test_batch_size=100


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)


class CDSSM(nn.Module):
    def __init__(self):
        super(CDSSM, self).__init__()
        # layers for query
        self.query_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.query_sem = nn.Linear(K, L)
        # layers for docs
        self.doc_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.doc_sem = nn.Linear(K, L)
        # learning gamma
        self.learn_gamma = nn.Conv1d(1, 1, 1)
    def forward(self, q, pos, negs):
        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
        q = q.transpose(1,2)
        #print("1:{}".format(q.shape))
        q_c = F.tanh(self.query_conv(q))
        #print("2:{}".format(q_c.shape))
        #print(q_c)
        # Next, we apply a max-pooling layer to the convolved query matrix.
        q_k = kmax_pooling(q_c, 2, 1)
        #print("3:{}".format(q_k.shape))
        q_k = q_k.transpose(1,2)
        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s • v + b_s). Again,
        # the paper does not include bias units.
        q_s = F.tanh(self.query_sem(q_k))
        #print("4:{}".format(q_s.shape))
        #print(q_s)
        q_s = q_s.resize(batch_size,L)
        #print("5:{}".format(q_s.shape))
        #print(q_s)
        # # The document equivalent of the above query model for positive document
        pos = pos.transpose(1,2)
        pos_c = F.tanh(self.doc_conv(pos))
        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1,2)
        pos_s = F.tanh(self.doc_sem(pos_k))
        pos_s = pos_s.resize(batch_size,L)
        # # The document equivalent of the above query model for negative documents
        negs = [neg.transpose(1,2) for neg in negs]
        neg_cs = [F.tanh(self.doc_conv(neg)) for neg in negs]
        neg_ks = [kmax_pooling(neg_c, 2, 1) for neg_c in neg_cs]
        neg_ks = [neg_k.transpose(1,2) for neg_k in neg_ks]
        neg_ss = [F.tanh(self.doc_sem(neg_k)) for neg_k in neg_ks]
        neg_ss = [neg_s.resize(batch_size,L) for neg_s in neg_ss]
        #print("6:{}".format(neg_ss.shape))
        
        #print('similarity')
        q_np=q_s.data.numpy()
        pos_np=pos_s.data.numpy()
        
        dots=[]
        for i in range(len(q_np)):
            dots.append([])
            dot = np.dot(q_np[i],pos_np[i])
            dot=Variable(torch.from_numpy(np.array(dot)))
            dots[i].append(dot)
        #print(dots)
        #print(neg_ss)
        
        for i in range(len(q_np)):
            for j in range(J):
                neg_ss[j][i].data.numpy()
                #print(neg_ss[j][i])
                dot = np.dot(q_np[i],neg_ss[j][i].data.numpy())
                dot=torch.from_numpy(np.array(dot))
                dots[i].append(dot)
        #print("dots")
        #print(dots)
        probs=[]
        # dots is a list as of now, lets convert it to torch variable
        for i in range(batch_size):
            do=dots[i]
            do = torch.stack(do)
            #print('after stack"{}'.format(do))
            
        # In this step, we multiply each dot product value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.
            with_gamma = self.learn_gamma(do.resize(J+1, 1, 1))
        # Finally, we use the softmax function to calculate P(D+|Q).
            prob = F.softmax(with_gamma)
            probs.append(prob)
        #print(probs)
        return probs

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
train_data=dataset.MusicSet('data_train.txt')
trainloader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
test_data=dataset.MusicSet('data_test.txt')
testloader = DataLoader(dataset=test_data, batch_size=test_batch_size,shuffle=True)

# model
model = CDSSM()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# output variable, remember the cosine similarity with positive doc was at 0th index
y = np.ndarray(batch_size)
# CrossEntropyLoss expects only the index as a long tensor
y[:] = 0
y = Variable(torch.from_numpy(y).long())

for epoch in range(nums_epoch):
    model.train()
    for batch_idx, (Q, P, N) in enumerate(trainloader):
        #Q, P, N = Q.to(device), P.to(device), N.to(device)
        model=model.double()
        #y_pred_list=[]
        y_pred = model(Q, P, N)
        #print(y_pred)
        n_correct=0
        for i in range(batch_size):
            record=y_pred[i].resize(1,J+1).float()
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
        acc=float(n_correct)/batch_size
        #print(acc)
        #print(a)
        loss = criterion(a, y) #Input&target
        #print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            '''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            '''
            print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.6f}, Acc: {:.6f}" 
                   .format(epoch+1, nums_epoch, batch_idx+1, len(trainloader), loss.item(),acc))
