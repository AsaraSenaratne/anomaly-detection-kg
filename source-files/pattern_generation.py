import seaborn as sns;
sns.set_theme(color_codes=True)
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import math
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.patches as mpatches
import seaborn as sns; sns.set_theme(color_codes=True)
import csv
from statistics import median

global ds, name, svm_output_file, support_file, tile_plot_file_name, association_plot_file_name
ds = 100


name = "YAGO Links"
svm_output_file = "../results/yago_links_svm_output.pkl"
support_file = "../results/yago_links_support.csv"
tile_plot_file_name = "../results/tile_plot_yago_links.eps"
association_plot_file_name = "../results/ass_plot_wikidata_yago_links.eps"
columns_start_count = 0

def identify_consistent_features():
    print("Identifying consistency of the features...")
    df = pd.read_pickle(svm_output_file)
    cols_extracted = {}
    for col in df.columns[columns_start_count:-2]:
        print(col)
        two_col_df = df[[col, "svm_binary_output"]]
        two_col_df['counter'] = 1
        groups = two_col_df.groupby([col, "svm_binary_output"])[
            'counter'].sum().reset_index()  # this removes gaps in the dataframe
        FA, FN, TA, TN = 0,0,0,0
        for val in groups[col].unique():
            if val == 0:
                sub_groups = groups.loc[groups[col] == val]
                for index, row in sub_groups.iterrows():
                    if row["svm_binary_output"] == -1:
                        FA = row["counter"]
                    else:
                        FN = row["counter"]
            else:
                sub_groups = groups.loc[groups[col] == val]
                for index, row in sub_groups.iterrows():
                    if row["svm_binary_output"] == -1:
                        TA = row["counter"]
                    else:
                        TN = row["counter"]
        consistency_score = abs((TN/(TN+FN)) - (TA/(TA+FA)))
        consistency_score = round(consistency_score, 4)
        cols_extracted[col] = consistency_score
    cols_extracted = dict(sorted(cols_extracted.items(), key=lambda item: item[1], reverse=True))
    print(cols_extracted)
    extract_consistent_features(df, cols_extracted)

def extract_consistent_features(df, cols_extracted):
    list_cols = []
    mean_consistency_score = sum(cols_extracted.values()) / len(cols_extracted.values())
    median_consistency_score = median(sorted(cols_extracted.values()))
    threshold_consistency_scores = max(mean_consistency_score, median_consistency_score)
    for key in cols_extracted.keys():
        if cols_extracted[key] >= threshold_consistency_scores:
            list_cols.append(key)
    list_cols.append("svm_binary_output")
    consistent_feature_df = df[list_cols]
    print(consistent_feature_df.columns)
    get_rows_with_plus_one_svm(consistent_feature_df, cols_extracted)

def get_rows_with_plus_one_svm(consistent_feature_df, cols_extracted):
    abnormal_group = consistent_feature_df.loc[consistent_feature_df["svm_binary_output"] == -1]
    add_patterns_col(abnormal_group, cols_extracted)

def add_patterns_col(abnormal_group, cols_extracted):
    col_pattern = []
    for index, row in abnormal_group.iterrows():
        pattern = ""
        for col in abnormal_group.columns[:-1]:
            pattern += str(row[col])
        col_pattern.append(pattern)
    abnormal_group["pattern"] = col_pattern
    cal_col_weight(abnormal_group, cols_extracted)

def cal_col_weight(abnormal_group, cols_extracted):
    col_weight = []
    for col in abnormal_group.columns[:-2]:
        count = len(abnormal_group[abnormal_group[col] == 1])
        weight = round(count / len(abnormal_group), 4)
        col_weight.append(weight)
    col_weight.append("NaN")
    col_weight.append("NaN")
    abnormal_group.loc["support"] = col_weight
    style_pattern_table(abnormal_group, cols_extracted)

def style_pattern_table(abnormal_group, cols_extracted):
    abnormal_group_last_row_rem = abnormal_group.iloc[:-1, :]
    unique_patterns = abnormal_group_last_row_rem['pattern'].unique()
    dict_pattern = {}
    for val in unique_patterns:
        count = len(abnormal_group_last_row_rem[abnormal_group_last_row_rem['pattern'] == val])
        dict_pattern[val] = count
    dict_pattern = dict(sorted(dict_pattern.items(), key=lambda item: item[1], reverse=True))
    cols = abnormal_group_last_row_rem.columns[:-2]
    pattern_dataframe = pd.DataFrame()
    pattern_dataframe["pattern_id"] = dict_pattern.keys()
    col_position = 0
    for val in cols:
        column_list = []
        for pattern in dict_pattern.keys():
            column_list.append(pattern[col_position])
        col_position += 1
        pattern_dataframe[val] = column_list
    pattern_dataframe["pattern_occurence"] = dict_pattern.values()
    pattern_dataframe = pattern_dataframe.append(abnormal_group.iloc[-1, :-2])
    styled = (pattern_dataframe.style.applymap(lambda v: 'background-color:blue' if v == '1' else '' 'background-color:red' if v == '0' else ''))
    styled.to_excel("assets/styled_pattern_table.xlsx")
    expand_kmeans_visualize(pattern_dataframe, cols_extracted)

def expand_kmeans_visualize(df, cols_extracted):
    # expanding the dataframe
    dataframe = df.iloc[:-1, :]
    columns = dataframe.columns
    expanded_numerical_table = pd.DataFrame(columns=columns)
    k = 0
    for index, rows in dataframe.iterrows():
        for i in range(0, int(rows["pattern_occurence"])):
            expanded_numerical_table.loc[k] = list(rows)
            k += 1

    # method call to create the association graph
    create_association_graph(expanded_numerical_table)

    # implementation of agglomerative clustering
    if len(dataframe) > 5:
        print("Conducting agglomerative clustering...")
        clustering_df = pd.DataFrame(columns=["pattern", "pattern_occurence"])
        clustering_df["pattern"] = df["pattern_id"].iloc[:-1]
        clustering_df["pattern_occurence"] = df["pattern_occurence"].iloc[:-1]
        clustering_df['pattern'] = clustering_df['pattern'].astype(str)
        clustering_df.to_csv("agglomerative_clustering.csv")
        clustering_df = clustering_df.sort_values(['pattern'])

        patterns_list, patterns_dict = [], {}
        for item in list(clustering_df["pattern"]):
            unique_cluster = []
            for element in item:
                unique_cluster.append(int(element))
            patterns_list.append([unique_cluster])
            row = clustering_df.loc[clustering_df['pattern'] == item]
            patterns_dict[item] = int(row["pattern_occurence"])

        while True:
            if len(patterns_list) <= 5:
                cluster_dict, leading_patterns_dict = {}, {}
                for cluster in patterns_list:
                    sum_pattern_occurence, pattern, max_pattern_occurrence, max_pattern = 0, "", 0, ""
                    for item in cluster:
                        binary_pattern = ""
                        for i in range(0, len(item)):
                            binary_pattern += str(item[i])

                        row = clustering_df.loc[clustering_df['pattern'] == binary_pattern]
                        if int(row["pattern_occurence"]) > max_pattern_occurrence:
                            max_pattern_occurrence = int(row["pattern_occurence"])
                            max_pattern = binary_pattern
                        sum_pattern_occurence += int(row["pattern_occurence"])
                        pattern += binary_pattern + ", "
                    leading_patterns_dict[max_pattern] = max_pattern_occurrence
                    pattern = pattern.rstrip(", ")
                    cluster_dict[pattern] = sum_pattern_occurence
                clustered_df = pd.DataFrame(cluster_dict.items(), columns=['pattern', 'pattern_occurence'])
                clustered_df.to_csv('agglomerative_clustering.csv', mode='a', header=True)
                break
            else:
                max_sim, c1, c2, c3, c4, c5, pattern_summation1, pattern_summation2 = -1, [], [], [], [], [], 0, 0
                for i in range(0, len(patterns_list) - 1):
                    for j in range(i + 1, len(patterns_list)):
                        avg_sim = cluster_avrg_sim(patterns_list[i], patterns_list[j])
                        if avg_sim > max_sim:
                            max_sim = avg_sim
                            c1 = patterns_list[i]
                            c2 = patterns_list[j]
                        elif avg_sim == max_sim:
                            summation_of_counts_1 = find_pattern(c1, c2, patterns_dict)
                            summation_of_counts_2 = find_pattern(patterns_list[i], patterns_list[j], patterns_dict)
                            if summation_of_counts_1 > summation_of_counts_2:
                                c1 = patterns_list[i]
                                c2 = patterns_list[j]
                patterns_list.append(c1 + c2)
                patterns_list.remove(c1)
                patterns_list.remove(c2)
        draw_mosaic(df, clustering_df, cluster_dict, leading_patterns_dict, cols_extracted)
    else:
        clustering_df = pd.DataFrame(columns=["pattern", "pattern_occurence"])
        clustering_df["pattern"] = df["pattern_id"].iloc[:-1]
        clustering_df["pattern_occurence"] = df["pattern_occurence"].iloc[:-1]
        clustering_df['pattern'] = clustering_df['pattern'].astype(str)
        clustering_df.to_csv("agglomerative_clustering.csv")
        clustering_df = clustering_df.sort_values(['pattern'])
        print(clustering_df)
        cluster_dict = {}
        for index, row in clustering_df.iterrows():
            cluster_dict[row['pattern']] = row['pattern_occurence']
        print(cluster_dict)
        draw_mosaic(df, clustering_df, cluster_dict, cluster_dict, cols_extracted)

def draw_mosaic(df, clustering_df, cluster_dict, leading_patterns_dict, cols_extracted):
    print("Plotting the mosaic plot...")
    print(df.columns)
    dataframe = df.iloc[:, 1:-1]
    tot_abnormal_records = sum(cluster_dict.values())
    columns_for_mosaic = dataframe.columns
    data, labels, props, support, col_color = {}, {}, {}, {}, {}

    # color palette used in the mosaic plot
    pal_hls = sns.color_palette("Blues", n_colors=6).as_hex()

    for pos_col, col in enumerate(columns_for_mosaic):
        for pos_pattern, pattern in enumerate(leading_patterns_dict.keys()):
            keys_list = list(cluster_dict)
            original_key = keys_list[pos_pattern]
            key = original_key.split(",")
            count_of_trues = 0
            for item in key:
                item = item.strip()
                if item[pos_col] == '1':
                    row = clustering_df.loc[clustering_df['pattern'] == item]
                    count_of_trues += int(row["pattern_occurence"])
            percentage_on_tile = float(count_of_trues / cluster_dict[original_key])

            # get the cluster size of the cluster representative
            cluster_size = [cluster_dict[key] for key in cluster_dict.keys() if pattern in key]
            data[(pattern, col)] = (cols_extracted[col] * math.log(cluster_size[0]+1))


            # set the color of the tiles
            if percentage_on_tile == 1.00:
                color = pal_hls[0]
            elif percentage_on_tile >= 0.75:
                color = pal_hls[1]
            elif percentage_on_tile >= 0.5:
                color = pal_hls[2]
            elif percentage_on_tile >= 0.25:
                color = pal_hls[3]
            elif percentage_on_tile > 0.0:
                color = pal_hls[4]
            else:
                color = pal_hls[5]

            # color = col_color[support[col]]
            props[(pattern, col)] = {'color': color}

            # set label to show on tile. No tile labels shown
            labels[(pattern, col)] = " "

    data = dict(sorted(data.items(), key=lambda item: item[1]))

    dict_1, dict_2, dict_3, dict_4, dict_5, keys_list = {}, {}, {}, {}, {}, []
    for key in data.keys():
        keys_list.append(key[1])
    keys_list = set(keys_list)
    for item in keys_list:
        temp_dict = {}
        for key in data.keys():
            if item == key[1]:
                temp_dict[key] = data[key]
                if len(temp_dict) == 5:
                    temp_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1]))
                    dict_1[list(temp_dict.keys())[0]] = temp_dict[list(temp_dict.keys())[0]]
                    dict_2[list(temp_dict.keys())[1]] = temp_dict[list(temp_dict.keys())[1]]
                    dict_3[list(temp_dict.keys())[2]] = temp_dict[list(temp_dict.keys())[2]]
                    dict_4[list(temp_dict.keys())[3]] = temp_dict[list(temp_dict.keys())[3]]
                    dict_5[list(temp_dict.keys())[4]] = temp_dict[list(temp_dict.keys())[4]]
                    break
    dict_1 = dict(sorted(dict_1.items(), key=lambda item: item[1], reverse=True))
    dict_2 = dict(sorted(dict_2.items(), key=lambda item: item[1], reverse=True))
    dict_3 = dict(sorted(dict_3.items(), key=lambda item: item[1], reverse=True))
    dict_4 = dict(sorted(dict_4.items(), key=lambda item: item[1], reverse=True))
    dict_5 = dict(sorted(dict_5.items(), key=lambda item: item[1], reverse=True))

    key_pair_dict, modified_dict = {}, {}
    for dictionary in [dict_1, dict_2, dict_3, dict_4, dict_5]:
        for key in dictionary.keys():
            key1 = key[0].replace("0", "F")
            key2 = key1.replace("1", "T")
            list_uppercase_index = [idx for idx in range(len(key[1])) if key[1][idx].isupper()]
            if len(list_uppercase_index) > 2:
                if list(dictionary.keys()).index(key) % 2 != 0:
                    if list_uppercase_index[1] == list_uppercase_index[2] - 2:
                        xtick_label = key[1][0:list_uppercase_index[1]] + "\n" + key[1][list_uppercase_index[1]:]
                    elif list_uppercase_index[1] == list_uppercase_index[2] - 1:
                        xtick_label = key[1][0:list_uppercase_index[1]] + "\n" + key[1][list_uppercase_index[1]:]
                    else:
                        xtick_label = key[1][0:list_uppercase_index[2]] + "\n" + key[1][list_uppercase_index[2]:]
                else:
                    if list_uppercase_index[1] == list_uppercase_index[2] - 2:
                        xtick_label = key[1][0:list_uppercase_index[1]] + "\n" + key[1][
                                                                                 list_uppercase_index[1]:] + "\n" + "\n"
                    elif list_uppercase_index[1] == list_uppercase_index[2] - 1:
                        xtick_label = key[1][0:list_uppercase_index[1]] + "\n" + key[1][list_uppercase_index[1]:] + "\n" + "\n"
                    else:
                        xtick_label = key[1][0:list_uppercase_index[2]] + "\n" + key[1][
                                                                                 list_uppercase_index[2]:] + "\n" + "\n"
            else:
                xtick_label = key[1][0:list_uppercase_index[1]] + "\n" + key[1][
                                                                         list_uppercase_index[1]:] + "\n" + "\n"

            key_pair_dict[key] = (key2, xtick_label)
            modified_dict[(key2, xtick_label)] = dictionary[key]
    data = modified_dict

    new_props = {}
    for key in key_pair_dict.keys():
        for key_props in props.keys():
            if key == key_props:
                new_props[key_pair_dict[key]] = props[key_props]
    props = new_props

    new_labels = {}
    for key in key_pair_dict.keys():
        for key_labels in labels.keys():
            if key == key_labels:
                new_labels[key_pair_dict[key]] = labels[key_labels]
    labels = new_labels

    labelizer = lambda k: labels[k]

    # handles of the color palette
    legend_labels = ["100%", "75% - 99%", "50% - 74%", "25% - 49%", "1% - 24%", "0%"]
    handles, i = [], 0
    for key in pal_hls:
        handle = mpatches.Patch(color=key, label=legend_labels[i])
        i += 1
        handles.append(handle)

    # bbox_to_anchor = (how much to go along x-axis. Higher the further, how much to go along y-axis
    fig, ax = plt.subplots(constrained_layout=True)  # set constrained layout to true so nothing gets cropped
    mosaic(data, label_rotation=0, title="", horizontal=False, properties=props, labelizer=labelizer, gap=0.01,ax=ax)
    fig.legend(handles=handles, bbox_to_anchor=(1.16, 0.8), loc='upper right', borderaxespad=0., fontsize=9)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=5)
    plt.grid(False)
    fig.savefig(tile_plot_file_name, bbox_inches='tight')

def create_association_graph(df):
    print("Plotting the association plot...")
    length_df = len(df)
    columns_list = list(df.iloc[:, 1:-1].columns)
    with open(support_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Col A", "Col B", "Support A", "Support notA", "Support B", "Support notB", "Support AB",
                         "BinaryCheck", "Support notAB", "BinaryCheck",
                         "Support AnotB", "BinaryCheck", "Support notAnotB", "BinaryCheck",
                         "MaximumSupportCriteria", "MaximumSupport"])
        for i in range(0, len(columns_list) - 1):
            for j in range(i + 1, len(columns_list)):
                supportA, supportnA, supportB, supportnB, supportAB, supportnAB, supportAnB, supportnAnB = 0, 0, 0, 0, 0, 0, 0, 0
                for index, row in df.iterrows():
                    if row[columns_list[i]] == '1':
                        supportA += 1
                    if row[columns_list[i]] == '0':
                        supportnA += 1
                    if row[columns_list[j]] == '1':
                        supportB += 1
                    if row[columns_list[j]] == '0':
                        supportnB += 1
                    if row[columns_list[i]] == '1' and row[columns_list[j]] == '1':
                        supportAB += 1
                    if row[columns_list[i]] == '0' and row[columns_list[j]] == '1':
                        supportnAB += 1
                    if row[columns_list[i]] == '1' and row[columns_list[j]] == '0':
                        supportAnB += 1
                    if row[columns_list[i]] == '0' and row[columns_list[j]] == '0':
                        supportnAnB += 1
                supportA = round(supportA / length_df, 4)
                supportnA = round(supportnA / length_df, 4)
                supportB = round(supportB / length_df, 4)
                supportnB = round(supportnB / length_df, 4)
                supportAB = round(supportAB / length_df, 4)
                supportnAB = round(supportnAB / length_df, 4)
                supportAnB = round(supportAnB / length_df, 4)
                supportnAnB = round(supportnAnB / length_df, 4)
                max_support = max(supportAB, supportnAB, supportAnB, supportnAnB)
                support_dict = {"supportAB": supportAB, "supportnAB": supportnAB, "supportAnB": supportAnB,
                                "supportnAnB": supportnAnB}
                writer.writerow([columns_list[i], columns_list[j], supportA, supportnA, supportB, supportnB,
                                 supportAB, (supportAB <= supportA and supportAB <= supportB),
                                 supportnAB, (supportnAB <= supportnA and supportnAB <= supportB),
                                 supportAnB, (supportAnB <= supportA and supportAnB <= supportnB),
                                 supportnAnB, (supportnAnB <= supportnA and supportnAnB <= supportnB),
                                 list(support_dict.keys())[list(support_dict.values()).index(max_support)],
                                 max_support])
    support_df = pd.read_csv(support_file)
    support_df = support_df.sort_values(['MaximumSupport'], ascending=False)
    support_df.to_csv(support_file)

    feature_support_criteria_dict, feature_support_dict, features_normal, features_flipped = {}, {}, {}, {}
    words_to_remove_dict = {"Same": "Diff", "Present": "Absent", "Match": "Mismatch", "High": "Low", "Valid":"Invalid",
                            "Common": "Uncom", "Before": "After", "Complete": "Incomp", "Inc": "Dec", "Freq":"Rare"}
    for index, row in support_df[:10].iterrows():
        feature_support_criteria_dict[(row["Col A"], row["Col B"])] = row["MaximumSupportCriteria"]
        feature_support_dict[(row["Col A"], row["Col B"])] = row["MaximumSupport"]
        features_normal[row["Col A"]] = row["Support A"]
        features_normal[row["Col B"]] = row["Support B"]

    for feature in features_normal.keys():
        for word in words_to_remove_dict.keys():
            if word in feature:
                new_feature = feature.replace(word, words_to_remove_dict[word])
                features_flipped[new_feature] = 1 - float(features_normal[feature])

    flipped_feature_support_dict = {}
    for key in feature_support_criteria_dict.keys():
        if feature_support_criteria_dict[key] == "supportAB":
            flipped_feature_support_dict[key] = feature_support_dict[key]
        if feature_support_criteria_dict[key] == "supportnAB":
            new_features_list = []
            for word in words_to_remove_dict.keys():
                if word in key[0]:
                    new_feature = key[0].replace(word, words_to_remove_dict[word])
                    new_features_list.append(new_feature)
            new_features_list.append(key[1])
            new_key = tuple(new_features_list)
            flipped_feature_support_dict[new_key] = feature_support_dict[key]
        if feature_support_criteria_dict[key] == "supportAnB":
            new_features_list = []
            new_features_list.append(key[0])
            for word in words_to_remove_dict.keys():
                if word in key[1]:
                    new_feature = key[1].replace(word, words_to_remove_dict[word])
                    new_features_list.append(new_feature)
            new_key = tuple(new_features_list)
            flipped_feature_support_dict[new_key] = feature_support_dict[key]
        if feature_support_criteria_dict[key] == "supportnAnB":
            new_features_list = []
            for item in key:
                for word in words_to_remove_dict.keys():
                    if word in item:
                        new_feature = item.replace(word, words_to_remove_dict[word])
                        new_features_list.append(new_feature)
            new_key = tuple(new_features_list)
            flipped_feature_support_dict[new_key] = feature_support_dict[key]

    G = nx.Graph()
    dict_edges = {k: v for k, v in
                  sorted(flipped_feature_support_dict.items(), key=lambda item: item[1], reverse=True)}

    # calculating node support
    adjusted_nodes_dict = {}
    features_normal.update(features_flipped)
    for key in features_normal.keys():
        adjusted_nodes_dict[key] = key + "\n" + str(int(float(features_normal[key]) * 100)) + "%"


    # matching nodes and edges to find common features
    list_edge_features = []
    for key in dict_edges.keys():
        list_edge_features.append(key[0])
        list_edge_features.append(key[1])
    list_edge_features = list(set(list_edge_features))
    for key in list(adjusted_nodes_dict):
        if key not in list_edge_features:
            adjusted_nodes_dict.pop(key)

    # adding edges in the graph
    for key in dict_edges.keys():
        G.add_edge(key[0], key[1], support=dict_edges[key])

    fig, ax = plt.subplots(constrained_layout=True)
    pos = nx.spring_layout(G)
    edge_labels = {(key[0], key[1]): str(int((float(dict_edges[key]) * 100))) + "%" for key in dict_edges.keys()}
    nx.draw_networkx(G, pos, with_labels=False, font_size=10, node_size=900, node_color="#FFFF00", edge_color="blue",
                     ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green', font_size=10, ax=ax,
                                 font_weight="bold")

    new_data_dict = {}
    print(adjusted_nodes_dict)
    for key in adjusted_nodes_dict.keys():
        print(key)
        list_uppercase_index = [idx for idx in range(len(key)) if key[idx].isupper()]
        print(list_uppercase_index)
        if len(list_uppercase_index) > 2:
            if list_uppercase_index[1] == list_uppercase_index[2] - 2:
                xtick_label = adjusted_nodes_dict[key][0:list_uppercase_index[1]] + "\n" + adjusted_nodes_dict[key][
                                                                                           list_uppercase_index[1]:]
            elif list_uppercase_index[1] == list_uppercase_index[2] - 1:
                xtick_label = adjusted_nodes_dict[key][0:list_uppercase_index[1]] + "\n" + adjusted_nodes_dict[key][
                                                                                           list_uppercase_index[1]:]
            else:
                xtick_label = adjusted_nodes_dict[key][0:list_uppercase_index[2]] + "\n" + adjusted_nodes_dict[key][
                                                                                           list_uppercase_index[2]:]

        else:
            xtick_label = adjusted_nodes_dict[key][0:list_uppercase_index[1]] + "\n" + adjusted_nodes_dict[key][
                                                                                       list_uppercase_index[1]:]
        new_data_dict[key] = xtick_label
    G = nx.relabel_nodes(G, new_data_dict)
    nx.draw_networkx_labels(G, pos, new_data_dict, font_size=10, ax=ax, font_weight="bold")


    plt.axis('off')
    l, r = plt.xlim()
    t, b = plt.ylim()
    plt.xlim(l - 0.4, r + 0.5)
    plt.ylim(t - 0.1, b + 0.2)
    fig.savefig(association_plot_file_name)

def hamming_dist(vec1, vec2):
    l1, l2 = len(vec1), len(vec2)
    assert l1 == l2, (l1, l2)  # Check both vectors have same length
    if (vec1 == vec2):  # Quick test the two vectors are the same
        return 0
    hd = 0
    for (i, elem1) in enumerate(vec1):
        if (elem1 != vec2[i]):
            hd += 1
    assert (hd > 0) and (hd <= l1), hd
    return hd

def cluster_avrg_sim(clus1, clus2):
    # Similarity is calculated as 1-(len(vec)-Hamming_distance)/len(vec)
    sim_list = []  # Keep all similarities
    l1 = len(clus1[0])  # Get length of first vector per cluster
    l2 = len(clus2[0])
    assert l1 == l2, (l1, l2)
    for vec1 in clus1:
        for vec2 in clus2:
            sim = 1.0 - float(hamming_dist(vec1, vec2)) / l1
            assert 0 <= sim <= 1.0, sim
            sim_list.append(sim)
    avr_sim = sum(sim_list) / len(sim_list)
    assert 0 <= avr_sim <= 1.0, avr_sim
    return avr_sim

def find_pattern(item1, item2, patterns_dict):
    sum_pattern_occurence = 0
    for cluster in [item1, item2]:
        pattern = ""
        for item in cluster:
            binary_pattern = ""
            for i in range(0, len(item)):
                binary_pattern += str(item[i])

            sum_pattern_occurence += int(patterns_dict[binary_pattern])

    return sum_pattern_occurence


identify_consistent_features()