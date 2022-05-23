import sys
import numpy as np
from matplotlib.pyplot import imread
import pickle
import os
import matplotlib.pyplot as plt

data_path = os.path.join('data/')
train_folder = os.path.join(data_path,'images_background')
valpath = os.path.join(data_path,'images_evaluation')

save_path = 'data/'

lang_dict = {}


def loadimgs(path,n=0):
    if not os.path.exists(path):
        print("unzipping")
        os.chdir(data_path)
        os.system("unzip {}".format(path+".zip" ))
    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n

    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)

        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))

            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X,y,lang_dict

X,y,c=loadimgs(train_folder)


with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    pickle.dump((X,c),f)


X,y,c=loadimgs(valpath)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
	pickle.dump((X,c),f)