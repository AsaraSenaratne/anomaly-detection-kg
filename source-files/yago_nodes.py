from rdflib import Graph, RDF, Literal, RDFS, plugin, OWL, XSD, SKOS, PROV
plugin.register('json-ld', 'Serializer', 'rdfextras.serializers.jsonld', 'JsonLDSerializer')
import csv
import pandas as pd
from collections import Counter
import networkx as nx
from nltk.tag import StanfordNERTagger
import spacy
import wikipediaapi
import jellyfish as jf
import json
import validators
import re


csv_links = "../results/yago-links.csv"
csv_node_labels = "../results/yago-node-labels.csv"
csv_nodes = "../results/yago-nodes.csv"
entity_file = "../results/entities_nodes.csv"
data_type_json = "../results/data-type-validation.json"


############################Triple Features############################################################################

def pred_occur():
#counts the no. of times a predicate occurs within the entire dataset
    print("inside pred_occur")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_node_labels['CountPredOccur'] = df_node_labels['predicate'].map(Counter(list(df_node_labels['predicate'])))
    df_node_labels.to_csv(csv_node_labels, index=False)

def subj_pred_occur():
# counts the no. of times a predicate occurs within the entity's group
    print("inside subj_pred_occur")
    df_node_labels = pd.read_csv(csv_node_labels)
    total_groups = df_node_labels.groupby(['subject', 'predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_node_labels['CountPredOccurofSubject'] = df_node_labels.set_index(['subject', 'predicate']).index.map(label_count.get)
    df_node_labels.to_csv(csv_node_labels, index=False)

def dup_triples():
# counts the no. of times a predicate occurs within the entity's group
    print("inside dup_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    total_groups = df_node_labels.groupby(['subject', 'predicate', 'object'])
    label_count = {}
    for group in total_groups.groups:
        try:
            label_count[group] = len(total_groups.get_group(group))
        except:
            continue
    df_node_labels['CountDupTriples'] = df_node_labels.set_index(['subject', 'predicate', 'object']).index.map(label_count.get)
    df_node_labels.to_csv(csv_node_labels, index=False)

def cal_haslabel_similarity():
# Calculate the similarity between two labels given a has_label predicate
    print("inside cal_haslabel_similarity")
    df_node_labels = pd.read_csv(csv_node_labels)
    label_count = {}
    for index, row in df_node_labels.iterrows():
        if row['predicate'] == "has_label":
            entity_split = row['subject'].split("/")[-1]
            underscore_removed = entity_split.replace("_", " ")
            wordnet_removed = underscore_removed.replace('wordnet', "")
            wikicat_removed = wordnet_removed.replace('wikicategory', "")
            subject_final = "".join(filter(lambda x: not x.isdigit(), wikicat_removed))
            subject_final = subject_final.strip()
            print(subject_final)
            similarity = jf.jaro_distance(subject_final, row['object'])
            label_count[(row['subject'], row['object'])] = round(similarity,2)
        else:
            label_count[(row['subject'], row['object'])] = "na"
    df_node_labels['SimSubjectObject'] = df_node_labels.set_index(['subject', 'object']).index.map(label_count.get)
    df_node_labels.to_csv(csv_node_labels, index=False)

def entity_recognition():
    print("inside entity_recognition")
    web_sm = spacy.load(r'/Users/asara/Documents/data_enrichment/assets/en_core_web_sm-3.0.0/en_core_web_sm/en_core_web_sm-3.0.0')
    st = StanfordNERTagger('assets/stanford-ner-4.2.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                           'assets/stanford-ner-4.2.0/stanford-ner-4.2.0.jar', encoding='utf-8')
    df_node_labels = pd.read_csv(csv_node_labels)
    dict_entity_type = {}
    entities = set(list(df_node_labels['subject']) + list(df_node_labels['object']))
    total_entities = len(entities)
    print("total_entities: " ,total_entities)
    count=0
    with open(entity_file, "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["entity", "entity_type"])
        for entity in entities:
            try:
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
                replacements = {"ORG": "ORGANIZATION", "GPE": "LOCATION", "LOC": "LOCATION"}
                replacer = replacements.get  # For faster gets.
                entity_types = [replacer(n, n) for n in entity_types]
                dict_entity_type[entity] = set(entity_types)
                writer.writerow([entity, set(entity_types)])
            except:
                dict_entity_type[entity] = "{O}"
                writer.writerow([entity, set(entity_types)])
    df_node_labels['SubjectEntityType'] = df_node_labels.set_index(['subject']).index.map(dict_entity_type.get)
    df_node_labels['ObjectEntityType'] = df_node_labels.set_index(['object']).index.map(dict_entity_type.get)
    df_node_labels.to_csv(csv_node_labels, index=False)

def pred_entity_type_occur():
#this method counts the no. of times a particular predicate occurs with the two given entity types of subject and object
    print("inside pred_entity_type_occur")
    df_node_labels = pd.read_csv(csv_node_labels)
    total_groups = df_node_labels.groupby(['SubjectEntityType','ObjectEntityType','predicate'])
    label_count = {}
    for group in total_groups.groups:
        try:
            label_count[group] = len(total_groups.get_group(group))
        except:
            continue
    df_node_labels['CountPredOccurEntityType'] = df_node_labels.set_index(['SubjectEntityType','ObjectEntityType','predicate']).index.map(label_count.get)
    df_node_labels.to_csv(csv_node_labels, index=False)

def validate_literal_data_type():
    print("inside validate_literal_data_type")
    df_node_labels = pd.read_csv(csv_node_labels)
    json_file = open(data_type_json,)
    data_type_dict = json.load(json_file)
    print(data_type_dict)
    validity_score = []
    for index, row in df_node_labels.iterrows():
        pred_extracted = row['predicate']
        if validators.url(pred_extracted):
            pred_extracted = row['predicate'].split("/")[-1]
        validity = "na"
        for key in data_type_dict.keys():
            if key in pred_extracted.lower():
                data_type = data_type_dict[key]
                try:
                    if data_type == "url":
                        validity = is_url(row['object'])
                        break
                    if data_type == "date":
                        validity = is_date(row['object'])
                        break
                    if data_type == 'integer':
                        validity = row['object'].isnumeric()
                        break
                    if data_type == 'time':
                        validity = re.match('\d{2}:\d{2}:\d{2}', row['object'])
                        break
                    if data_type == 'string':
                        validity = ((not row['object'].isnumeric()) and (not row['object'] == "") and (not validators.url(row['object'])\
                                                                                                       and (not is_date(row['object']))))
                        break
                except:
                    validity = 0
        validity_score.append(validity)
    df_node_labels['ValidityOfLiteral'] = validity_score
    df_node_labels.to_csv(csv_node_labels, index=False)


############################Node Features############################################################################
def tot_literals():
#count total number of literal based triples an entity has
    print("inside tot_predicates")
    df_node_labels = pd.read_csv(csv_node_labels)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        try:
            label_count[group] = len(total_groups.get_group(group))
        except:
            continue
    data = {'node':list(label_count.keys()), 'CountLiterals': list(label_count.values()) }
    df_nodes = pd.DataFrame.from_dict(data)
    df_nodes.to_csv(csv_nodes, index=False)

def count_dif_literal_types():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_dif_literal_types")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        print(fetched_group)
        count_dif_literals = len(fetched_group['predicate'].unique())
        label_count[group] = count_dif_literals
    df_nodes['CountDifLiteralTypes'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def count_isa_triples():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_isa_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        print(fetched_group)
        count_dif_literals = len(fetched_group[fetched_group['predicate']=='isa'])
        label_count[group] = count_dif_literals
    df_nodes['CountIsaPredicate'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def count_haslabel_triples():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_haslabel_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        print(fetched_group)
        count_dif_literals = len(fetched_group[fetched_group['predicate']=='has_label'])
        label_count[group] = count_dif_literals
    df_nodes['CountHaslabelPredicate'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)


def count_subclassof_triples():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_subclassof_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        print(fetched_group)
        count_dif_literals = len(fetched_group[fetched_group['predicate']=='subClassOf'])
        label_count[group] = count_dif_literals
    df_nodes['CountSubclassofPredicate'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def count_subpropertyof_triples():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_subpropertyof_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        print(fetched_group)
        count_dif_literals = len(fetched_group[fetched_group['predicate']=='subPropertyOf'])
        label_count[group] = count_dif_literals
    df_nodes['CountSubpropofPredicate'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def count_high_sim_labels():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_high_sim_labels")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        count = 0
        for index, row in fetched_group.iterrows():
            if row['SimSubjectObject'] != 'na' and float(row['SimSubjectObject']) >=0.5:
                count+=1
        label_count[group] = count
    df_nodes['CountHighSimLabels'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def count_low_sim_labels():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_low_sim_labels")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        count = 0
        for index, row in fetched_group.iterrows():
            if row['SimSubjectObject'] != 'na' and float(row['SimSubjectObject']) <0.5:
                count+=1
        label_count[group] = count
    df_nodes['CountLowSimLabels'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def tot_outgoing_links():
    print("inside tot_incoming_links")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    df_links = pd.read_csv(csv_links)
    groups_in_node_labels = df_node_labels.groupby(['subject'])
    groups_in_links = df_links.groupby(['subject'])
    label_count = {}
    for group in df_nodes['node']:
        try:
            fetched_node_label_groups = len(groups_in_node_labels.get_group(group))
        except:
            fetched_node_label_groups = 0
        try:
            fetched_link_groups = len(groups_in_links.get_group(group))
        except:
            fetched_link_groups = 0
        label_count[group] = fetched_node_label_groups + fetched_link_groups
    df_nodes['OutDegree'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def tot_incoming_links():
    print("inside tot_outgoing_links")
    df_nodes = pd.read_csv(csv_nodes)
    df_links = pd.read_csv(csv_links)
    groups_in_links = df_links.groupby(['object'])
    label_count = {}
    for group in df_nodes['node']:
        try:
            fetched_link_groups = len(groups_in_links.get_group(group))
        except:
            fetched_link_groups = 0
        label_count[group] = fetched_link_groups
    df_nodes['InDegree'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def find_com_rare_entity_type():
# counts the no. of times a predicate occurs within the entity's group
    print("inside find_com_rare_entity_type")
    df_node_links = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_links.groupby(['subject'])
    entity_count_node_max, entity_count_node_min = {}, {}
    for group in total_groups.groups:
        entity_count = {}
        sub_group = total_groups.get_group(group).groupby(['SubjectEntityType','ObjectEntityType'])
        for entity_group in sub_group.groups:
            entity_count[entity_group] = len(sub_group.get_group(entity_group))
        key_max = max(entity_count.keys(), key=(lambda k: entity_count[k]))
        key_min = min(entity_count.keys(), key=(lambda k: entity_count[k]))
        entity_count_node_max[group] = [key_max]
        entity_count_node_min[group] = [key_min]
    df_nodes['CommonPredType'] = df_nodes.set_index(['node']).index.map(entity_count_node_max.get)
    df_nodes['RarePredType'] = df_nodes.set_index(['node']).index.map(entity_count_node_min.get)
    df_nodes.to_csv(csv_nodes, index=False)

def count_occur_dup_triples():
# counts the no. of duplicate triples a node has got
    print("inside count_occur_dup_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        count=0
        for index, row in total_groups.get_group(group).iterrows():
            if row['CountDupTriples'] > 1:
                count+=row['CountDupTriples']
        label_count[group] = count
    df_nodes['CountDupTriples'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

def count_invalid_literals():
    print("inside count_invalid_literals")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        count=0
        for index, row in total_groups.get_group(group).iterrows():
            if row['ValidityOfLiteral'] == False:
                count+=1
        label_count[group] = count
    df_nodes['CountInvalidTriples'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)

########################################Special Functions###################################
def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    from dateutil.parser import parse
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def is_url(url):
#check for valid URL
    if not validators.url(url):
        return False
    else:
        return True


pred_occur()
subj_pred_occur()
dup_triples()
cal_haslabel_similarity()
tot_literals()
count_dif_literal_types()
count_isa_triples()
count_haslabel_triples()
count_subclassof_triples()
count_subpropertyof_triples()
count_high_sim_labels()
count_low_sim_labels()
tot_outgoing_links()
tot_incoming_links()
validate_literal_data_type()
count_occur_dup_triples()
count_invalid_literals()
entity_recognition()
pred_entity_type_occur()
find_com_rare_entity_type()

