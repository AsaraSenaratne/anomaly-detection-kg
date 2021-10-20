# Unsupervised Anomaly Detection in Knowledge Graphs

## Introduction
This repository provides the implementation of an approach to unsupervised feature-based approach to anomaly detection in knowledge graphs. We first 
characterize triples in a directed edge-labelled knowledge graph using a set of binary features, and then employ a one-class support vector machine classifier to classify these
triples  as normal or abnormal.  After selecting the features that have the highest consistency with the SVM outcomes, we provide a visualization of the
identified anomalies, and the list of anomalous triples, thus supporting non-technical domain experts to understand the anomalies present in a knowledge graph.

## Technologies
This project is implemented using:
* Python 3.6

Following Python packages are used in the projet. 
* pandas v1.3.1
* nltk v3.6.2
* networkx v2.6.2
* Wikipedia_API v0.5.4
* numpy v1.21.2
* rdflib v6.0.0
* jellyfish v0.8.8
* seaborn v0.11.2
* validators v0.18.2
* matplotlib v3.4.2
* statsmodels v0.13.0
* spacy v3.1.3
* parameters v0.2.1
* python_dateutil v2.8.2
* scikit_learn v1.0
* pyunpack v0.2.2

## Execute the project
Clone the project:
```
git clone https://github.com/AsaraSenaratne/anomaly_detection_kg
```

Install the required packages:
```
pip install -r requirements.txt
```

Open a terminal and change directory to the cloned project:
```
cd <path_to_directory>/abnormality_detection_kg/source-files

```
Ensure that the following libraries inside the 'assets' folder are properly
downloaded properly, and imported within yago_links.py.
````
en_core_web_sm-3.0.0
stanford-ner-4.2.0
````
Run main.py:
```
python3 -W ignore main.py
```

## Folder structure of the project
Assets folder - this folder contains the datasets used for analysis.  
Results folder - this folder gets created at program execution. All the plots, pickle files and CSV files that are generated as a result of program execution will be stored in this folder.  
Source folder - source code files are within this folder.  
Other files:
* LICENSE - contains information of the license under which this project is available.
* README - provides the users with an introduction to the project and guidance to use.
* requirements.txt - text file with information about the packages used in this project.

## Reading the results folder
The results folder contains all graphs and other intermediary results files generated at different steps of program execution. CSV files are available to view the initial nodes dataset, any intermediary supporting datasets built (such as merged datasets) and SVM output.  
Visualization outputs are available as a tile plot and an association plot in .eps format.  
