import graph_tool as gt
from graph_tool.all import *
from scipy.sparse import load_npz


def load(path):
    g = Graph()
    digraph_coo = load_npz(path)
    out_node_list = digraph_coo.row
    in_node_list = digraph_coo.col
    for i in range(out_node_list.shape[0]):
        # print(out_node_list[i], in_node_list[i])
        g.add_edge(out_node_list[i], in_node_list[i], True)
    return g


def plot_tree(g, output):
    return gt.draw.graphviz_draw(g, layout="dot", output=output)


def plot_gv(inpath, outpath):
    #  node[shape = point]
    with open(outpath, "w") as fout:
        fout.write("digraph g {\n")
        nodes = set()
        digraph_coo = load_npz(inpath)
        out_node_list = digraph_coo.row
        in_node_list = digraph_coo.col
        for i in range(out_node_list.shape[0]):
            nodes.add(out_node_list[i])
            nodes.add(in_node_list[i])
            fout.write("%s -> %s;\n" % (out_node_list[i], in_node_list[i]))
        for p in nodes:
            fout.write("%s[shape=point];\n" % p)
        fout.write("}")


fname = "graphs13/balanced_tree/branching=10-log_num_nodes=13-transitive_closure=False/2150935259.npz"

plot_gv(fname, fname + ".gv")
