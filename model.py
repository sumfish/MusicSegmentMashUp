import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

WORD_DEPTH = 128 # See equation (1).
# Uncomment it, if testing
# WORD_DEPTH = 1000
K = 128 # Dimensionality of the max-pooling layer. See section 3.4.
K2 = 128
L = 64 # Dimensionality of latent semantic space. See section 3.5.
J = 2 # Number of random unclicked documents serving as negative examples for a query. See section 4.
# kernel size of time(word_depth*time)
FILTER_LENGTH = 3 # We only consider one time step for convolutions.

def kmax_pooling(x, dim, k):
    ### get max vlaue across time of feature map
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)


class CDSSM(nn.Module):
    def __init__(self):
        super(CDSSM, self).__init__()
        # layers for query
        self.query_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.query_conv1 = nn.Conv1d(K, K2, FILTER_LENGTH)
        self.query_sem = nn.Linear(K2, L)
        # layers for docs
        self.doc_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.doc_conv1 = nn.Conv1d(K, K2, FILTER_LENGTH)
        self.doc_sem = nn.Linear(K2, L)
        # learning gamma
        self.learn_gamma = nn.Conv1d(1, 1, 1)

    def forward(self, q, pos, negs):
        size=len(q)
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
        q = q.transpose(1,2) #[N, WORD_DEPTH(12 or 128), query_len(100)]
        #print("1:{}".format(q.shape))

        q_c = torch.tanh(self.query_conv(q)) #[N, K, query_len(98)]
        #print("1:{}".format(q_c.shape))

        q_c = torch.tanh(self.query_conv1(q_c)) #[N, K2, query_len(96)]
        #print("2:{}".format(q_c.shape))

        # Next, we apply a max-pooling layer to the convolved query matrix.
        q_k = kmax_pooling(q_c, 2, 1) #[N, K2, 1]
        #print("3:{}".format(q_k.shape))

        q_k = q_k.transpose(1,2) #[N, 1, K2]
        #print("3:{}".format(q_k.shape))

        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s • v + b_s). 
        q_s = torch.tanh(self.query_sem(q_k)) #[N, 1, L]
        #print("4:{}".format(q_s.shape))
        #print(q_s)

        q_s = q_s.resize(size,L) #[N,L]
        #print("5:{}".format(q_s.shape)) 
        #print(q_s)

        # # The document equivalent of the above query model for positive document
        pos = pos.transpose(1,2)
        pos_c = torch.tanh(self.doc_conv(pos))
        pos_c = torch.tanh(self.doc_conv1(pos_c))
        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1,2)
        pos_s = torch.tanh(self.doc_sem(pos_k))
        pos_s = pos_s.resize(size,L)

        # # The document equivalent of the above query model for negative documents
        negs = [neg.transpose(1,2) for neg in negs]
        neg_cs = [torch.tanh(self.doc_conv(neg)) for neg in negs]
        neg_cs = [torch.tanh(self.doc_conv1(neg)) for neg in neg_cs]
        neg_ks = [kmax_pooling(neg_c, 2, 1) for neg_c in neg_cs]
        neg_ks = [neg_k.transpose(1,2) for neg_k in neg_ks]
        neg_ss = [torch.tanh(self.doc_sem(neg_k)) for neg_k in neg_ks]
        neg_ss = [neg_s.resize(size,L) for neg_s in neg_ss]
        #print("6:{}".format(neg_ss.shape))
        
        '''
        cos=nn.CosineSimilarity(dim=1)
        outputs=cos(q_s,pos_s)
        outputs.resize(-1,1)
        print(outputs.shape)
        print(outputs)
        '''
        #print('similarity')
        q_np=q_s.cpu().detach().numpy()
        #pos_np=pos_s.data.numpy()
        pos_np=pos_s.cpu().detach().numpy()

        dots=[]
        #[[tensor],[tenosor],[]]
        for i in range(size):
            dots.append([])
            dot = np.dot(q_np[i],pos_np[i])
            dot=Variable(torch.from_numpy(np.array(dot)))
            dots[i].append(dot)
        #print(dots)
        
        for i in range(size):
            for j in range(J):
                #neg_ss[j][i].data.numpy()
                #print(neg_ss[j][i])
                dot = np.dot(q_np[i],neg_ss[j][i].cpu().detach().numpy())
                dot=torch.from_numpy(np.array(dot))
                dots[i].append(dot)
        #print("dots")
        #print(dots)
        probs=[]
        # dots is a list as of now, lets convert it to torch variable
        for i in range(size):
            do=dots[i]
            do = torch.stack(do)
            #print('after stack"{}'.format(do))
            
        # In this step, we multiply each dot product value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.
            with_gamma = self.learn_gamma(do.resize(J+1, 1, 1))
        # Finally, we use the softmax function to calculate P(D+|Q).
            #print(with_gamma)
            prob = F.softmax(with_gamma)
            probs.append(prob)
        #print(probs)
        return probs