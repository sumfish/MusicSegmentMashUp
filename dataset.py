from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import numpy as np
from PIL import Image

music_folder='./better_music_segments/'
#data_path='./audio_docu/'
data_path='./better_music_segments_docu/'
filename_txt='filenames.txt'
label_txt='labels.txt'
neg_nums=2 ## can adjust 
#avgv = np.load(data_path + 'avg.npy')
#stdv = np.load(data_path + 'std.npy')

def default_audio_loader(data, S_max):
    y, sr = librosa.core.load(music_folder+data, sr=22050)
    S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    #S = librosa.feature.chroma_stft(y=y, sr=sr,
    #       n_chroma=12, n_fft=4096)
    #print(S.shape)#(128, N)
    #print('S={}'.format(S))
    if S.shape[-1] > S_max:
        S = S[:, :S_max]
        #print(S.shape)
    else:
        S = np.pad(S, ((0,0), (0, max(0, S_max-S.shape[-1]))),'constant', constant_values=(0))
    #print(S.shape) #(128, S_max)
    #S = np.transpose(np.log(1+10000*S))
    #S = np.transpose(np.log(1+1000*S))
    S = S.transpose()
    #print(S.shape) #(S_max, 128)
    #S = (S-avgv)/stdv
    #S = np.expand_dims(S, 2) # (N, 128, 1)
    #print(S.shape)
    return S

'''  
def get_feature(data):
    #print(data)
    y, sr = librosa.load(music_folder+data)
    mel=librosa.feature.melspectrogram(y=y, sr=sr)
    
    librosa.display.specshow(librosa.power_to_db(mel,
                                                ref=np.max),
                                                y_axis='mel', fmax=8000,
                                                x_axis='time')
    plt.show()
    
    return mel
'''

class MusicSet(Dataset):
    S_max = 100
    def __init__(self,data_txt,feature=default_audio_loader):
        ### confirm imge path
        seeds=[]
        negs=[]
        audio_path=[]
        label=[]
        
        fopen1=open(data_path+data_txt,'r')
        for line in fopen1:
            words=line.split('[')
            q_words=words[0].split()
            seeds.append(q_words[0])
            #print(words[1])
            #input()
            words[1]=words[1].strip('\n')
            #print(words[1])
            words[1]=words[1].strip()
            #print(words[1])
            neg_word=words[1].split(',')
            neg_word[neg_nums]=neg_word[neg_nums].strip(']')
            neg_list=[]
            for i in range(5):
                neg_list.append(neg_word[i])
            negs.append(neg_list)
            #print(negs)
        fopen1.close()

        fopen2=open(data_path+filename_txt,'r')
        lines = fopen2.readlines() 
        for i in range(len(lines)):
            audio_path.append(lines[i].strip('\n'))
        fopen2.close()
        
        fopen3=open(data_path+label_txt,'r')
        linesl = fopen3.readlines() 
        for i in range(len(linesl)):
            label.append(linesl[i].strip('\n'))
        #print(label)
        fopen3.close()
        
        #print(seeds)  
        #print(negs)      
        self.seeds = seeds # now&next
        self.negs = negs
        self.audio_path = audio_path
        self.label = label
        self.feature = feature
    

    def __getitem__(self, index):
        
        seed = int(self.seeds[index])
        pos = seed+1
        neg = self.negs[index]
        
        ## feature representation
        seed_arr=self.feature(self.audio_path[seed],self.S_max)
        pos_arr=self.feature(self.audio_path[pos],self.S_max)
        #print(seed_arr.shape)
        if(neg_nums==1):
            neg_arr=self.feature(self.audio_path[int(neg[0])],self.S_max)
            return seed_arr, pos_arr, neg_arr
        else:
            neg_set=[]
            for i in range(neg_nums):
                neg_arr=self.feature(self.audio_path[int(neg[i])],self.S_max)
                neg_set.append(neg_arr)
                
        return seed_arr, pos_arr, neg_set

    def __len__(self):
        return len(self.seeds)


#test_data=MusicSet('data_test.txt')
#train_data=MusicSet('datas_train.txt').__getitem__(0)
'''
trainloader = DataLoader(dataset=train_data, batch_size=16,shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=16)


print(type(trainloader))

for index, q in enumerate(trainloader):
        print(index)
        print(len(q))
        #print(len(n))
'''
