def params(dataset):
    if dataset == 1:
        # YAGO Links
        params.feature_file = "../results/features_selected_yago_links.csv"
        params.svm_output_pkl = "../results/yago_links_svm_output.pkl"
        params.initial_attribute_count = 19
        params.name = "YAGO Links"
        params.support_file = "../results/yago_links_support.csv"
        params.tile_plot_file_name = "../results/tile_plot_yago_links.eps"
        params.association_plot_file_name = "../results/ass_plot_wikidata_yago_links.eps"
        params.columns_start_count = 0
        params.ds = 100

    elif dataset == 2:
        # YAGO nodes
        params.feature_file = "../results/features_selected_yago_nodes.csv"
        params.svm_output_pkl = "../results/yago_nodes_svm_output.pkl"
        params.initial_attribute_count = 8
        params.name = "YAGO Nodes"
        params.support_file = "../results/yago_nodes_support.csv"
        params.tile_plot_file_name = "../results/tile_plot_yago_nodes.eps"
        params.association_plot_file_name = "../results/ass_plot_wikidata_yago_nodes.eps"
        params.columns_start_count = 0
        params.ds = 100

    else:
        print("Invalid dataset requested")