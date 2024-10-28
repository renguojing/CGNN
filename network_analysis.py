import networkx as nx
import numpy as np

source_path = 'data/twitter1_youtube/twitter3.txt'
target_path = 'data/twitter1_youtube/youtube3.txt'
anchor_path = 'data/twitter1_youtube/groundtruth'
# test_path = 'data/four-twit/ind.four_twit.anchors.test'
out_path = 'data/twitter1_youtube/statistics.txt'

def interop(g_s, g_t, anchor):
    dict = {x[0]: x[1] for x in anchor}
    count = 0
    for edge in g_s.edges():
        
        if dict.__contains__(edge[0]) and dict.__contains__(edge[1]) and g_t.has_edge(dict[edge[0]],dict[edge[1]]):
            count += 1
    out = count * 2 / (len(g_s.edges()) + len(g_t.edges()))
    return out

# s_edge = np.loadtxt(source_path, dtype=int, delimiter=' ')
# print(len(s_edge), np.max(s_edge))
g_s = nx.read_edgelist(source_path, nodetype=int, delimiter='\t')
g_t = nx.read_edgelist(target_path, nodetype=int, delimiter='\t')
anchor = np.loadtxt(anchor_path, dtype=int, delimiter=' ')
# test_anchor = np.loadtxt(test_path, dtype=int, delimiter='\t')
# anchor = np.vstack((anchor, test_anchor))

with open(out_path, 'w+') as f:
    print('Nodes of source network: ', len(g_s.nodes()), file=f)
    print('Edges of source network: ',len(g_s.edges()), file=f)
    print('Nodes of target network: ',len(g_t.nodes()), file=f)
    print('Edges of target network: ',len(g_t.edges()), file=f)
    print('Anchor links: ',anchor.shape[0], file=f)
    print('Interoperability: ', interop(g_s, g_t, anchor), file=f)
    f.close()