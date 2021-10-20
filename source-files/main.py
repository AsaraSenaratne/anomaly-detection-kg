import os
import time
import requests
import py7zr
import zipfile
import yago_links as yl
import yago_nodes as nl
import svm_training as svm
import pattern_generation as pg
import tarfile

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
with open("../assets/stanford-ner-4.2.0.zip", 'w') as stanford:
    stanford.write(response.content)

print("Extracting Stanford NER 4.2.0")
with zipfile.ZipFile("../assets/stanford-ner-4.2.0.zip", 'w') as stan:
    stan.extractall(path='../assets')

print("Downloading en_core_web_sm-3.0.0")
response = requests.get("https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz")
with open("../assets/en_core_web_sm-3.0.0.tar.gz", 'w') as encore:
    encore.write(response.content)

print("Extracting en_core_web_sm-3.0.0")
fname = "../assets/en_core_web_sm-3.0.0.tar.gz"
tar = tarfile.open(fname, "r:gz")
tar.extractall()

def generate_features():
    print("-----------------------------------")
    print("Generating features for YAGO-1")
    start_time = time.time()
    yl.construct_df()
    execution_time = time.time() - start_time
    print('Total runtime taken feature generation: %.6f sec' % (execution_time))
    learn_one_class_svm()

def learn_one_class_svm():
    print("-----------------------------------")
    print("Started learning one-class SVM")
    start_time = time.time()
    svm.get_abnormal_counts()
    execution_time = time.time() - start_time
    print('Total runtime taken feature generation: %.6f sec' % (execution_time))
    visualization()

def visualization():
    print("-----------------------------------")
    print("Started learning one-class SVM")
    start_time = time.time()
    pg.identify_consistent_features()
    execution_time = time.time() - start_time
    print('Total runtime taken feature generation: %.6f sec' % (execution_time))

generate_features()



