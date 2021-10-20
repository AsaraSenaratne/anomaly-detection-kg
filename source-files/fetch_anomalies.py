import pandas as pd
from collections import Counter
import csv

def get_user_input():
    dataset_path = input("Enter absolute file path of SVM output: ")
    print("")
    print("  1 - Print the nodes with anomalies \n ",
          "2 - Print the triples with anomalies \n ",
          "3 - Fetch the triples given the anomalous pattern \n ",
          "4 - Fetch the triples given a feature \n ",
          )
    operation_number = int(input("Enter the number relevant to the operation you wish to perform from the above list: "))
    fetch_data_from_file(dataset_path, operation_number)

def fetch_data_from_file(dataset_path,operation_number):
    df = pd.read_csv(dataset_path)
    df = df.iloc[:, :-1]
    df_abnormal_all = df.loc[df['svm_binary_output'] == -1]

    if operation_number == 1:
        subject_nodes = list(df_abnormal_all["subject"])
        target_nodes = list(df_abnormal_all["object"])
        all_nodes = Counter(subject_nodes+target_nodes)
        sorted_all_nodes = {k: v for k, v in sorted(all_nodes.items(), key=lambda item: item[1], reverse=True)}
        print(sorted_all_nodes)
        print("{:<10} {:<10}".format('Node', 'No. of anomalous triples involved'))
        for node, value in sorted_all_nodes.items():
            print("{:<10} {:<10}".format(node, value))

    elif operation_number == 2:
        print(df_abnormal_all)

    elif operation_number == 3:
        dataset_name = input("Enter the dataset name: ")
        features = input("Enter the anomalous feature names seperated by a comma: ")
        features = features.split(",")
        feature_patterns = input("Enter the anomalous feature patterns seperated by a comma: ")
        feature_patterns = feature_patterns.split(",")
        replacements = {"F": 0, "T": 1}
        replacer = replacements.get
        file_name = "anomalous_patterns"+dataset_name+".csv"
        with open(file_name,'a') as features_files:
            writer = csv.writer(features_files)
            for feature_pattern in feature_patterns:
                writer.writerow(feature_pattern)
                feature_pattern_text = [replacer(n, n) for n in feature_pattern]
                feature_pattern_split = [char for char in feature_pattern_text]
                patterns_and_columns = dict(zip(features, feature_pattern_split))
                for column in patterns_and_columns.keys():
                    df_abnormal_all = df_abnormal_all[df_abnormal_all[column] == patterns_and_columns[column]]
                df_abnormal_all.to_csv(file_name, mode='a', header=True)

    elif operation_number == 4:
        print("")

    else:
        print("Invalid operation number entered")


get_user_input()
# fetch_data_from_file()