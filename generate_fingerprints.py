import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import time
import numpy as np
import os
import json



def gen_random_data(BATCH = 1):
    X = torch.randn((BATCH, 3, 256, 256), dtype=torch.float32)
    #X = X.unsqueeze(0)
    return X

def preprocess_image(npy_dir, preprocess):
    arr = np.load(npy_dir)
    arr = np.divide(arr, 255)
    #shape H x W x 3, np array
    arr = torch.Tensor(arr)
    arr = arr.permute(2, 0, 1)
    #shape 3 x H x W, torch tensor
    prep_arr = preprocess(arr)
    prep_arr = prep_arr.unsqueeze(0)
    return prep_arr

def handle_dir(dir, model, dir_tag = 'SAMPLE'):


    print('Fingerprint gen for %s'%(dir.split('/')[-2]))
    savedir = '/Users/seancondon/Desktop/065_final/fingerprints/'
    savedir += dir.split('/')[-2]
    savedir += '.json'
    outDict = {}
    preprocess_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    done = 0
    for f in os.listdir(dir):
        if f.endswith('.npy'):
            done+=1
            filename = int(f.split('.')[0])
            process_dir = os.path.join(dir, f)
            preprocessed = preprocess_image(process_dir, preprocess=preprocess_transform)
            out = model(preprocessed)
            out = out[0].detach().numpy()
            outlist = [float(i) for i in out]
            outDict[filename] = outlist
            if done %50 == 0:
                print('at step %d'%(done))


    with open(savedir, 'w+') as f:
        json.dump(outDict, f)


#identity module to replace fc
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



if __name__ == '__main__':

    #load in resnet18
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    #get rid of final fully connected layer
    model.fc = Identity()
    model.eval()


    #trial_dir = '/Users/seancondon/Desktop/065_final/ingested_images/2ef3a4994a_0CCD105428INSPIRE-ortho/'
    completed = os.listdir('/Users/seancondon/Desktop/065_final/fingerprints/')
    meta = '/Users/seancondon/Desktop/065_final/ingested_images/'
    for directory in os.listdir(meta):
        if directory != '.DS_Store' and (directory+'.json') not in completed:
            curr_dir =  meta + directory + '/'
            handle_dir(curr_dir, model)
        else:
            print('FAILED OR COMPLETED!')
            print(directory)




