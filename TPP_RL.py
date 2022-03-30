import networkx as nx

graph = nx.DiGraph()
nodes = [f"{i}" for i in range(1, 13)]
nodes.extend([chr(i) for i in range(1, 13)])
graph.add_nodes_from([])


class Node:
    def __init__(self, name, direct, sig):
        self.name = name
        self.direct = direct
        self.sig = sig


class Edge:
    def __init__(self, prev, to, tp, direct):
        self.prev = prev
        self.to = to
        self.tp = tp
        self.direct = direct
