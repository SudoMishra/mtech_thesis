import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
# direction nodes

G.add_node('D1',ntype="direction")
G.add_node('D2',ntype="direction")

# intermediate nodes

G.add_node('a',ntype="mid")
G.add_node('b',ntype="mid")
G.add_node('c',ntype="mid")
G.add_node('d',ntype="mid")
G.add_node('e',ntype="mid")

# Platform nodes

# left platform_node
G.add_node('1l',ntype="Platform")
G.add_node('2l',ntype="Platform")
G.add_node('3l',ntype="Platform")
# right platform_node
G.add_node('1r',ntype="Platform")
G.add_node('2r',ntype="Platform")
G.add_node('3r',ntype="Platform")

# Edges
G.add_edge('D1','a',etype="arr")
G.add_edge('D1','e',etype="arr")
G.add_edge('D1','b',etype="dep")
G.add_edge('a','1l',etype="arr")
G.add_edge('a','2l',etype="arr")
G.add_edge('e','2l',etype="arr")
G.add_edge('b','2l',etype="dep")
G.add_edge('b','3l',etype="dep")
G.add_edge('1r','c',etype="dep")
G.add_edge('2r','c',etype="dep")
G.add_edge('2r','d',etype="arr")
G.add_edge('3r','d',etype="arr")
G.add_edge('c','D2',etype="dep")
G.add_edge('d','D2',etype="arr")

# ncolors = {
#     "direction": "r",
#     "mid": "g",
#     "Platform": "b"
# }
ncolors = nx.get_node_attributes(G,"ntype")
print(ncolors)
options = {
    "node_size": 1500,
    "alpha":0.3,
    # "node_color": ncolors.values()
}

pos = nx.spring_layout(G,seed=42)
nx.draw(G,pos,**options)
plt.show()
# fig,ax = plt.subplots()

# nx.draw_networkx_edges(
#     G,
#     pos=pos,
#     ax=ax,
#     arrows=True,
#     arrowstyle="-",
#     min_source_margin=15,
#     min_target_margin=15,
# )

# tr_figure = ax.transData.transform
# tr_axes = fig.transFigure.inverted().transform
# plt.show()