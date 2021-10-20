from rdflib import Graph,FOAF,URIRef, RDF, Literal, RDFS, plugin
plugin.register('json-ld', 'Serializer', 'rdfextras.serializers.jsonld', 'JsonLDSerializer')
import csv
import pandas as pd
from collections import Counter
import networkx as nx
from nltk.tag import StanfordNERTagger
import spacy

#file names
csv_links = "../results/yago-links.csv"
csv_node_labels = "../results/yago-node-labels.csv"
entity_file = "../results/entities_links.csv"
csv_with_features = "../results/features_selected_yago_links.csv"

print("Populating the graph")
graph = Graph()
graph.parse('../assets/yago-1.0.0-turtle.ttl', format='ttl')

def construct_df():
    print("inside construct_df")
    with open(csv_links, 'w', encoding="utf-8") as yago_links:
        writer = csv.writer(yago_links)
        writer.writerow(["subject", "predicate", "object"])
        for statement in graph:
            if statement[1] == RDF.type or isinstance(statement[2], Literal):
                continue
            else:
                writer.writerow([statement[0], statement[1], statement[2]])

    with open(csv_node_labels, 'w', encoding="utf-8") as yago_nodes:
        writer = csv.writer(yago_nodes)
        writer.writerow(["subject", "predicate", "object"])
        for statement in graph:
            if statement[1] == RDF.type:
                writer.writerow([statement[0],"isa", statement[2]])
            elif statement[1]==RDFS.label and isinstance(statement[2], Literal) and RDFS.subClassOf:
                writer.writerow([statement[0],"has_label" , statement[2]])
            elif statement[1]==RDFS.subClassOf and isinstance(statement[2], Literal):
                writer.writerow([statement[0],"subClassOf" , statement[2]])
            elif statement[1]==RDFS.subPropertyOf and isinstance(statement[2], Literal):
                writer.writerow([statement[0],"subPropertyOf" , statement[2]])
            elif isinstance(statement[2], Literal):
                writer.writerow([statement[0],statement[1], statement[2]])
    del globals()['graph']
    entity_recognition()

def subject_out_degree():
    print("inside source_out_degree")
    df_links = pd.read_csv(csv_links)
    df_links['SubjectOutDeg'] = df_links['subject'].map(Counter(list(df_links['subject'])))
    df_links.to_csv(csv_links,index=False)
    subject_in_degree()

def subject_in_degree():
    print("inside source_in_degree")
    df_links = pd.read_csv(csv_links)
    df_links['SubjectInDeg'] = df_links['subject'].map(Counter(list(df_links['object'])))
    df_links.to_csv(csv_links, index=False)
    object_out_degree()

def object_out_degree():
    print("inside source_out_degree")
    df_links = pd.read_csv(csv_links)
    df_links['ObjectOutDeg'] = df_links['object'].map(Counter(list(df_links['subject'])))
    df_links.to_csv(csv_links, index=False)
    object_in_degree()

def object_in_degree():
    print("inside source_in_degree")
    df_links = pd.read_csv(csv_links)
    df_links['ObjectInDeg'] = df_links['object'].map(Counter(list(df_links['object'])))
    df_links.to_csv(csv_links, index=False)
    pred_occur()

def entity_recognition():
    print("inside entity_recognition")
    web_sm = spacy.load(r'../assets/en_core_web_sm-3.0.0/en_core_web_sm/en_core_web_sm-3.0.0')
    st = StanfordNERTagger('../assets/stanford-ner-4.2.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                           '../assets/stanford-ner-4.2.0/stanford-ner-4.2.0.jar', encoding='utf-8')
    df_links = pd.read_csv(csv_links)
    dict_entity_type = {}
    entities = set(list(df_links['subject']) + list(df_links['object']))
    total_entities = len(entities)
    print("total_entities: " ,total_entities)
    count=0
    with open(entity_file, "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["entity", "entity_type"])
        for entity in entities:
            count = count + 1
            print("entity number: ", count)
            print("more to go number: ", total_entities-count)
            entity_types = []
            entity_split = entity.split("/")[-1]
            underscore_removed = entity_split.replace("_", " ")
            wordnet_removed = underscore_removed.replace('wordnet', "")
            wikicat_removed = wordnet_removed.replace('wikicategory', "")
            entity_final = "".join(filter(lambda x: not x.isdigit(), wikicat_removed))
            entity_final = entity_final.strip()
            print(entity_final)
            spacy_ner = [(X.text, X.label_) for X in web_sm(entity_final).ents]
            if spacy_ner:
                for item in spacy_ner:
                    entity_types.append(item[1])
            else:
                stanford_ner = st.tag([entity_final])
                for item in stanford_ner + spacy_ner:
                    entity_types.append(item[1])
            replacements = {"ORG": "ORGANIZATION", "GPE": "LOCATION"}
            replacer = replacements.get  # For faster gets.
            entity_types = [replacer(n, n) for n in entity_types]
            dict_entity_type[entity] = set(entity_types)
            writer.writerow([entity, set(entity_types)])
    df_links['SubjectEntityType'] = df_links.set_index(['subject']).index.map(dict_entity_type.get)
    df_links['ObjectEntityType'] = df_links.set_index(['object']).index.map(dict_entity_type.get)
    df_links.to_csv(csv_links, index=False)
    entity_predicate_occur()

def entity_predicate_occur():
    print("inside entity_predicate_occur")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['SubjectEntityType', 'predicate', 'ObjectEntityType'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['EntityPredOccur'] = df_links.set_index(['SubjectEntityType', 'predicate', 'ObjectEntityType']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    subject_out_degree()

def pred_occur():
    print("inside pred_occur")
    df_links = pd.read_csv(csv_links)
    df_links['PredOccur'] = df_links['predicate'].map(Counter(list(df_links['predicate'])))
    df_links.to_csv(csv_links, index=False)
    subj_pred_occur()

def subj_pred_occur():
    print("inside subj_pred_occur")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['subject', 'predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['SubPredOccur'] = df_links.set_index(['subject','predicate']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)

def pred_obj_occur():
    print("inside pred_obj_occur")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['predicate', 'object'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['PredObjOccur'] = df_links.set_index(['predicate','object']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    corroborative_paths()

def subj_obj_occur():
    print("inside subj_obj_occur")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['subject', 'object'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['SubObjOccur'] = df_links.set_index(['subject','object']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    dup_triples()

def dup_triples():
    print("inside dup_triples")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['subject', 'predicate','object'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['DupTriples'] = df_links.set_index(['subject','predicate','object']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    pred_obj_occur()

def corroborative_paths():
    #this method checks for the existence of alternative knowledge paths.
    print("inside corroborative_paths")
    df_links = pd.read_csv(csv_links)
    label_count = {}
    G = nx.DiGraph()
    for index, rows in df_links.iterrows():
        G.add_edges_from([(rows["subject"],rows["object"])])
        print(rows["subject"],rows["object"])
    for index, rows in df_links.iterrows():
        print("inside corroborative paths")
        count = len(list(nx.all_simple_paths(G, rows["subject"], rows["object"], cutoff=3)))
        print(rows["subject"],rows["object"], count)
        label_count[(rows["subject"], rows["object"])] = count
    df_links['CorrPaths'] = df_links.set_index(['subject', 'object']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    pred_entity_type_occur()

def pred_entity_type_occur():
#this method counts the no. of times a particular predicate occurs with the two given entity types of subject and object
    print("inside pred_entity_type_occur")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['SubjectEntityType','ObjectEntityType','predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['PredEntityOccur'] = df_links.set_index(['SubjectEntityType','ObjectEntityType','predicate']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    pred_entity_type_occur_with_source()

def pred_entity_type_occur_with_source():
#this method counts the no. of times a particular predicate occurs with the subject entity types
    print("inside pred_entity_type_occur_with_source")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['SubjectEntityType','predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['PredEntityTypeWithSubject'] = df_links.set_index(['SubjectEntityType','predicate']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    pred_entity_type_occur_with_target()

def pred_entity_type_occur_with_target():
#this method counts the no. of times a particular predicate occurs with the object entity types
    print("inside pred_entity_type_occur_with_target")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['ObjectEntityType','predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['PredEntityTypeWithObject'] = df_links.set_index(['ObjectEntityType','predicate']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    freq_occur_entity()

def freq_occur_entity():
    print("inside freq_occur_entity")
    df_links = pd.read_csv(csv_links)
    for column in ['SubjectEntityType','ObjectEntityType']:
        new_col_name = "Count" + column
        df_links[new_col_name] = df_links[column].map(Counter(list(df_links[column])))
    df_links.to_csv(csv_links, index=False)
    feature_reduction()

def feature_reduction():
    print("Inside feature reduction")
    df = pd.read_csv(csv_links)
    df_fileterd = df.iloc[:,3:]
    for col in df_fileterd.columns:
        count_unique = len(df[col].unique())
        if count_unique == 1:
            print(col)
            df_fileterd.drop(col, inplace=True, axis=1)
    columns = list(df_fileterd.columns)
    corr_feature_list = []
    for i in range(0, len(columns)-1):
        for j in range(i+1, len(columns)):
            print(columns[i],columns[j])
            try:
                correlation = df_fileterd[columns[i]].corr(df_fileterd[columns[j]])
            except:
                correlation = 0
            if correlation == 1:
                print(columns[i], columns[j])
                corr_feature_list.append(columns[i])
                corr_feature_list.append(columns[j])
    remove_corr_features(corr_feature_list, df_fileterd, df)

def remove_corr_features(corr_feature_list, df_fileterd, df):
    print("Correlated Features: ", corr_feature_list)
    features_to_remove  = [input("Enter the features to remove seperated by a comma without spaces: ")]
    if features_to_remove[0] == '':
        gen_binary_feature(df_fileterd, df)
    else:
        for feature in features_to_remove:
            df_fileterd.drop(feature, inplace=True, axis=1)
        gen_binary_feature(df_fileterd, df)

def gen_binary_feature(df_fileterd, df):
    print("inside binary_features")
    columns = df_fileterd.columns
    for column in columns:
        new_col = []
        new_col_name = "Freq" + column
        for index, row in df_fileterd.iterrows():
            if row[column] > df_fileterd[column].median():
                new_col.append(1)
            else:
                new_col.append(0)
        df[new_col_name] = new_col
    df.to_csv(csv_with_features, index=False)

