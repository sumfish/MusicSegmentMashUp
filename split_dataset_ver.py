import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
#import librosa.display
import madmom
from pydub import AudioSegment
import math
import json

NEED_COPY_FOLDER = False
filename='_original.mp3'
pianofile='piano.mp3'
in_dir = 'D:/2019/Summer/intern/audio_word/pop2jazz/'
in_dir += 'transcription_with_disjoint_notes.from_separated.soft_jazz_trio/20190403_095002.20190524_082516'
cut_multitrack = True
folder_dict = {}

'''
def copy_folder_structure(inputpath, outputpath):
    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, os.path.relpath(dirpath, inputpath))
        if not os.path.isdir(structure): os.mkdir(structure)
        else: print("Folder does already exits!")
'''
def find_downbeats(source):
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[4, 4], fps=100)
    act = madmom.features.downbeats.RNNDownBeatProcessor()(source)
    a = proc(act)
    return a

def load_audio(source):
    sig, sr = librosa.core.load(source) # sr = 22050
    #sig = AudioSegment.from_file(source)
    #sound2 = AudioSegment.from_file(piano_file)
    #print('Time:{}, Length:{}'.format(len(sig)/sr, len(sig)))
    return sig, sr
    #return sig

def segment(source, pianotrack, outputpath, outputpath_p, index):#, sr=44100):
    # Pydub do things in sr=1000
    a = find_downbeats(source)
    print('Track downbeats done ...')
    #sig = load_audio(source)
    
    sig_p, sr_p = load_audio(pianotrack)
    if(cut_multitrack ==True):
        sig, sr = load_audio(source)
    print('Load audio done ...')
    #sigs = []
    start = 0
    #has_record=False
    for i, c in enumerate(a):
        if c[-1] == 1:
            
            #if (has_record==False):
            #    has_record=True
            #    start=int(c[0]*sr)
            #    continue
            output_name_p = os.path.join(outputpath_p,'cut{:03d}_{:03d}.wav'.format(index, int(i/4)+1))
            #print(output_name_p)
            #sigs.append(sig[start : int(i[0]*sr)])
            cut_p = sig_p[start : int(c[0]*sr_p)]
            #cut.export(output_name,format='wav')
            librosa.output.write_wav(output_name_p, cut_p, sr_p)

            if(cut_multitrack==True):
                output_name = os.path.join(outputpath,'cut{:03d}_{:03d}.wav'.format(index, int(i/4)+1))
                cut = sig[start : int(c[0]*sr)]
                librosa.output.write_wav(output_name, cut, sr)
            start = int(c[0]*sr_p)
            #librosa.output.write_wav(output_name, sigs[i], sr)
    #sigs.append(sig[start:])
    #print(len(sigs)) # num of bars

def dictToTxt(in_dict, out_txt):
    with open(out_txt, 'w') as file:
        file.write(json.dumps(in_dict))

def txtToDict(in_txt):
    with open(in_txt, 'r') as file:
        a = json.load(file)
    return a
###########################################
out_dir_piano='./music_segments_piano'
out_dir_all='./music_segments_all_track'
# if NEED_COPY_FOLDER:
#     copy_folder_structure(in_dir, out_dir)
#     NEED_COPY_FOLDER = False

index = 0
for f in os.listdir(in_dir):
    if os.path.isdir(os.path.join(in_dir,f)):
        print(f) #folder
        for ff in os.listdir(os.path.join(in_dir,f)):
            print('---',ff) #sub-folder
            fder = os.path.join(os.path.join(in_dir,f),ff)
            print(fder)
            if os.path.isdir(fder):
                folder_dict['{:03d}'.format(index)] = os.path.join(f+'/',ff)
                print(folder_dict)
                save_dir_piano = os.path.join(out_dir_piano, '{:03d}'.format(index))
                save_dir_all = os.path.join(out_dir_all, '{:03d}'.format(index))
                print(save_dir_piano)

                if not os.path.isdir(save_dir_piano): 
                    os.mkdir(save_dir_piano)
                if not os.path.isdir(save_dir_all): 
                    os.mkdir(save_dir_all)
                segment(os.path.join(fder, filename),os.path.join(fder, pianofile), save_dir_all, save_dir_piano, index)
                print('done')
                index += 1
                # for file in os.listdir(fder):
                #     print(file)

    dictToTxt(folder_dict, 'file.txt')

#d = txtToDict('file.txt')
#print(',,,',d)