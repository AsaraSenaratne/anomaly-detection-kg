import os
import time
import requests
import py7zr
import zipfile
import yago_links as yl
import yago_nodes as nl
import svm_training as svm
import pattern_generation as pg


if not os.path.isdir("../results"):
    os.mkdir("../results")

if not os.path.isdir("../assets"):
    os.mkdir("../assets")

print("Downloading KG")
response = requests.get('https://yago-knowledge.org/data/yago1/yago-1.0.0-turtle.7z')
with open("../assets/yago-1.0-turtle.7z", 'wb') as yago:
    yago.write(response.content)

print("Extracting the downloaded KG")
with py7zr.SevenZipFile('../assets/yago-1.0-turtle.7z', mode='r') as extyago:
    extyago.extractall(path='../assets')

print("Downloading Stanford NER 4.2.0")
response = requests.get('https://nlp.stanford.edu/software/stanford-ner-4.2.0.zip')
with zipfile.ZipFile("../assets/stanford-ner-4.2.0.zip", 'w') as stan:
    stan.extractall(path='../assets')


def generate_features():
    print("-----------------------------------")
    print("Generating features for YAGO-1")
    start_time = time.time()
    yl.construct_df()
    execution_time = time.time() - start_time
    print('Total runtime taken feature generation: %.6f sec' % (execution_time))

def learn_one_class_svm():
    print("-----------------------------------")
    print("Started learning one-class SVM")
    svm.get_abnormal_counts()



