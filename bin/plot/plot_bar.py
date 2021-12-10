# draw dimension vs f1

import csv
import json


def load_csv(filename, metric="AUC"):
    with open(filename) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.reader(csv_file, delimiter="\t", quotechar='"')
        line_count = 0
        key2id = {}
        graph2method2dim = {}
        for row in csv_reader:
            if line_count == 0:
                key2id = dict([(k, i) for i, k in list(enumerate(row))])
                # print(key2id)
                line_count += 1
                print(key2id.items())
            else:
                method_name = row[key2id["model_type"]]
                dim = int(row[key2id["dim"]])
                # print(row[key2id["graph info"]])
                # graph_info = json.loads(row[key2id["graph info"]])
                # graph_type = row[key2id["type"]]
                graph_type = "_".join(row[key2id["path"]].split("/")[2:4])
                print(row[key2id["path"]].split("/")[2:4])
                transitive_closure = row[key2id["transitive_closure"]]
                graph_type = graph_type + "_" + str(transitive_closure)
                if metric == "AUC":
                    metric = float(row[key2id["AUC"]])
                else:
                    metric = float(row[key2id["F1"]])

                if graph_type not in graph2method2dim:
                    graph2method2dim[graph_type] = {}
                if method_name not in graph2method2dim[graph_type]:
                    if method_name in ["box", "vector", "complex_vector"]:
                        graph2method2dim[graph_type][method_name] = {
                            4: [],
                            16: [],
                            64: [],
                        }
                    else:
                        graph2method2dim[graph_type][method_name] = {
                            8: [],
                            32: [],
                            128: [],
                        }

                graph2method2dim[graph_type][method_name][dim].append(metric)
    return graph2method2dim


import matplotlib.pyplot as plt
import numpy as np

# customize


def plot_me(graph_type, method_list, data, plot_legend=False):

    # if graph_type not in ["nested_chinese_restaurant_process_True",
    #                       "nested_chinese_restaurant_process_False",
    #                       "balanced_tree_True",
    #                       "balanced_tree_False",
    #                       "price_True",
    #                       "price_False",
    #                       "scale_free_network_False",
    #                       "kronecker_graph_False",
    #                      ]:
    #     raise Exception("character not exist")
    for method in method_list:
        if method not in [
            "vector",
            "bilinear_vector",
            "complex_vector",
            "oe",
            "poe",
            "box",
            "lorentzian_distance",
        ]:
            raise Exception("method not exist")

    x_axis = [3, 5, 7]
    x_ticks = x_axis  # [x for x in range(len(x_axis))]
    x_tick_labels = [2 ** x for x in x_axis]
    vector_y_axis_mean, vector_y_axis_std = [0, 0, 0], [0, 0, 0]
    for i, dim in enumerate([4, 16, 64]):
        vector_y_axis_mean[i] = np.mean(data[graph_type]["vector"][dim])
        vector_y_axis_std[i] = np.std(data[graph_type]["vector"][dim])

    bilinear_y_axis_mean, bilinear_y_axis_std = [0, 0, 0], [0, 0, 0]
    for i, dim in enumerate([8, 32, 128]):
        bilinear_y_axis_mean[i] = np.mean(data[graph_type]["bilinear_vector"][dim])
        bilinear_y_axis_std[i] = np.std(data[graph_type]["bilinear_vector"][dim])

    complex_y_axis_mean, complex_y_axis_std = [0, 0, 0], [0, 0, 0]
    for i, dim in enumerate([4, 16, 64]):
        complex_y_axis_mean[i] = np.mean(data[graph_type]["complex_vector"][dim])
        complex_y_axis_std[i] = np.std(data[graph_type]["complex_vector"][dim])

    oe_y_axis_mean, oe_y_axis_std = [0, 0, 0], [0, 0, 0]
    for i, dim in enumerate([8, 32, 128]):
        oe_y_axis_mean[i] = np.mean(data[graph_type]["oe"][dim])
        oe_y_axis_std[i] = np.std(data[graph_type]["oe"][dim])

    poe_y_axis_mean, poe_y_axis_std = [0, 0, 0], [0, 0, 0]
    for i, dim in enumerate([8, 32, 128]):
        poe_y_axis_mean[i] = np.mean(data[graph_type]["poe"][dim])
        poe_y_axis_std[i] = np.std(data[graph_type]["poe"][dim])

    box_y_axis_mean, box_y_axis_std = [0, 0, 0], [0, 0, 0]
    for i, dim in enumerate([4, 16, 64]):
        box_y_axis_mean[i] = np.mean(data[graph_type]["box"][dim])
        box_y_axis_std[i] = np.std(data[graph_type]["box"][dim])

    lorentzian_y_axis_mean, lorentzian_y_axis_std = [0, 0, 0], [0, 0, 0]
    for i, dim in enumerate([8, 32, 128]):
        lorentzian_y_axis_mean[i] = np.mean(
            data[graph_type]["lorentzian_distance"][dim]
        )
        lorentzian_y_axis_std[i] = np.std(data[graph_type]["lorentzian_distance"][dim])

    models_y_axis_mean = {
        "vector": vector_y_axis_mean,
        "bilinear_vector": bilinear_y_axis_mean,
        "complex_vector": complex_y_axis_mean,
        "oe": oe_y_axis_mean,
        "poe": poe_y_axis_mean,
        "box": box_y_axis_mean,
        "lorentzian_distance": lorentzian_y_axis_mean,
    }
    models_y_axis_std = {
        "vector": vector_y_axis_std,
        "bilinear_vector": bilinear_y_axis_std,
        "complex_vector": complex_y_axis_std,
        "oe": oe_y_axis_std,
        "poe": poe_y_axis_std,
        "box": box_y_axis_std,
        "lorentzian_distance": lorentzian_y_axis_std,
    }

    # left below code alone

    plt.rc("axes", titlesize=14)
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)
    plt.rc("legend", fontsize=11)

    #     STYLE_MAP = {
    #         "vector": {"color": "lime", "marker": ".", "markersize": 7, 'label': "Vector", 'linewidth': 1},
    #         "bilinear_vector": {"color": "darkgreen", "marker": "^", "markersize": 7, 'label': "Bilinear", 'linewidth': 1},
    #         "complex_vector": {"color": "green", "marker": "^", "markersize": 7, 'label': "Complex", 'linewidth': 1},
    #         "oe": {"color": "brown", "marker": "s", "markersize": 7, 'label': "OE", 'linewidth': 1},
    #         "poe": {"color": "darkorange", "marker": "s", "markersize": 7, 'label': "POE", 'linewidth': 1},
    #         "box": {"color": "red", "marker": "*", "markersize": 7, 'label': "Gumbel Box", 'linewidth': 1},
    #         "lorentzian_distance": {"color": "blue", "marker": "*", "markersize": 7, 'label': "Lorentzian", 'linewidth': 1},

    #     }
    #     FILL_MAP = {
    #         "vector": "lime",
    #         "bilinear_vector": "darkgreen",
    #         "complex_vector": "green",
    #         "oe": "brown",
    #         "poe": "darkorange",
    #         "box": "red",
    #         "lorentzian_distance": "blue"
    #     }

    # colors = ["#19ABDB", "#00C096","#545BC5", "#EFC87E", "#629968", "#41DB53", "#FB7C7F"]
    #              vector, bilinear, complex,     oe,       poe,       box,     lorentz
    # colors = ["#00C096", "#19ABDB","#545BC5", "#E34A2B", "#FA8650", "#EB265C", "#EBDE18"]
    # colors = ["#00C096", "#19ABDB","#545BC5", "#EB7399", "#FA8650", "#EB5500", "#EBDE18"]
    colors = [
        "#00C096",
        "#19ABDB",
        "#545BC5",
        "#E001D1",
        "#E00501",
        "#F07F00",
        "#F0D101",
    ]

    # colors = ["#DB804D", "#5BB0FF", "#2CE66B", "#EBEAB5", "#9942EB"]
    # STYLE_MAP = {
    #         "vector": {"color": colors[0], "marker": "s", "markersize": 7, 'label': 'Sim', 'linewidth': 1},
    #         "bilinear_vector": {"color": colors[1], "marker": "s", "markersize": 7, 'label': "Bilinear", 'linewidth': 1},
    #         "complex_vector": {"color": colors[2], "marker": "s", "markersize": 7, 'label': "Complex", 'linewidth': 1},
    #         "oe": {"color": colors[3], "marker": "s", "markersize": 7, 'label': "OE", 'linewidth': 1},
    #         "poe": {"color": colors[4], "marker": "s", "markersize": 7, 'label': "POE", 'linewidth': 1},
    #         "box": {"color": colors[5], "marker": "s", "markersize": 7, 'label': "Box", 'linewidth': 1},
    #         "lorentzian_distance": {"color": colors[6], "marker": "s", "markersize": 7, 'label': "Hyperbolic", 'linewidth': 1},
    #     }

    STYLE_MAP = {
        "vector": {
            "color": colors[0],
            "marker": "o",
            "markersize": 7,
            "label": "Sim",
            "linewidth": 1,
        },
        "bilinear_vector": {
            "color": colors[1],
            "marker": ".",
            "markersize": 7,
            "label": "Bilinear",
            "linewidth": 1,
        },
        "complex_vector": {
            "color": colors[2],
            "marker": "x",
            "markersize": 7,
            "label": "Complex",
            "linewidth": 1,
        },
        "oe": {
            "color": colors[3],
            "marker": "*",
            "markersize": 7,
            "label": "OE",
            "linewidth": 1,
        },
        "poe": {
            "color": colors[4],
            "marker": "^",
            "markersize": 7,
            "label": "POE",
            "linewidth": 1,
        },
        "box": {
            "color": colors[5],
            "marker": "s",
            "markersize": 7,
            "label": "Box",
            "linewidth": 1,
        },
        "lorentzian_distance": {
            "color": colors[6],
            "marker": "p",
            "markersize": 7,
            "label": "Hyperbolic",
            "linewidth": 1,
        },
    }

    FILL_MAP = {
        "vector": colors[0],
        "bilinear_vector": colors[1],
        "complex_vector": colors[2],
        "oe": colors[3],
        "poe": colors[4],
        "box": colors[5],
        "lorentzian_distance": colors[6],
    }
    patterns = {
        "vector": "//",
        "bilinear_vector": "\\\\",
        "complex_vector": "**",
        "oe": "OO",
        "poe": "oo",
        "box": "--",
        "lorentzian_distance": "..",
    }

    plt.gcf().clear()

    scale_ = 1.0
    new_size = (scale_ * 8, scale_ * 10)
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(new_size)

    # fig, ax = plt.subplots()

    for i, model_name in enumerate(
        method_list
    ):  # ["vector", "bilinear", "poe", "box"]:
        # x = np.log(np.array(models_x_axis[model_name])+1e-10)
        x = np.array(x_ticks)
        y_mean = np.array(models_y_axis_mean[model_name])
        y_std = np.array(models_y_axis_std[model_name])
        # ax.plot(x+i/5-0.6, y_mean, **STYLE_MAP[model_name])
        ## lines:
        x = x[np.isfinite(y_mean)]
        y_std = y_std[np.isfinite(y_mean)]
        y_mean = y_mean[np.isfinite(y_mean)]
        # if model_name == 'box':
        #     import pdb; pdb.set_trace()
        ax.plot(x, y_mean, **STYLE_MAP[model_name])
        plt.fill_between(
            x,
            y_mean - 0.5 * y_std,
            y_mean + 0.5 * y_std,
            alpha=0.3,
            color=FILL_MAP[model_name],
        )
        ## bars:
        # plt.errorbar(x, y_mean, y_std, 0.2, alpha=0.3, color=FILL_MAP[model_name])
        # ax.bar(x+i/5-0.6, y_mean, yerr=y_std*0.5, width=0.15, align='center', alpha=1.0, color=FILL_MAP[model_name], capsize=2, hatch=patterns[model_name], label=STYLE_MAP[model_name]['label'])
        ax.set_xticklabels(x_tick_labels)

    x_label = "Number of parameters per node"
    x_range = [2, 8]
    y_label = "F1"
    y_range = [0, 1]

    # plt.xlim(left=x_range[0], right=x_range[1])
    ax.set_xticks(x)
    # plt.set_xticks(x_pos)
    # plt.xlim(left=0, right=50)
    plt.ylim(bottom=y_range[0], top=y_range[1])
    plt.locator_params(axis="x", nbins=6)
    plt.xlabel(x_label, fontsize=25)
    if plot_legend:
        plt.ylabel(y_label, fontsize=25)
        # plt.legend(loc='upper left', fontsize=25)
        plt.legend(loc="lower right", fontsize=12)
    # plt.title(graph_type_to_tile(graph_type), fontsize=25)
    plt.title(graph_type_to_tile(graph_type), fontsize=12)
    # plt.show()
    plt.tight_layout()
    plt.savefig("/tmp/zzz_synth_result." + graph_type + ".pdf")
    plt.close()


# choices of graph types:


def graph_type_to_tile(gt):
    splt = [x.capitalize() for x in gt.split("_")]
    res = []
    for x in splt:
        if x == "True":
            res.append("(TC)")
        elif x != "False":
            res.append(x)
    return " ".join(res)


# "price_True", "price_False"
# "nested_chinese_restaurant_process_True","nested_chinese_restaurant_process_False",
# "balanced_tree_True", "balanced_tree_False",
# "scale_free_network_False", "kronecker_graph_False",

data = load_csv("results.tsv", "F1")

# row1 = ["balanced_tree_True", "balanced_tree_False", "nested_chinese_restaurant_process_True","nested_chinese_restaurant_process_False"]
# row2 = ["scale_free_network_False", "kronecker_graph_False", "price_True", "price_False"]

# for idx,g in enumerate(row1):
#     plot_me(g, ['vector', 'bilinear_vector','complex_vector','lorentzian_distance','oe','poe','box'], data, False)
# plot_me("kronecker_graph_False", ['vector', 'bilinear_vector','complex_vector','lorentzian_distance','oe','poe','box'], data, plot_legend)

# for idx,g in enumerate(row2):
#     plot_me(g, ['vector', 'bilinear_vector','complex_vector','lorentzian_distance','oe','poe','box'], data, idx == 0)

for gt in data.keys():
    print("gt: %s" % gt)
    plot_me(
        gt,
        [
            "vector",
            "bilinear_vector",
            "complex_vector",
            "lorentzian_distance",
            "oe",
            "poe",
            "box",
        ],
        data,
        True,
    )
