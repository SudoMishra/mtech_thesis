import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from collections import defaultdict


def read_arr_paths():
    arr_paths = []
    # dep_paths = []
    dir_names = ['D1', 'D2', 'D3', 'D4']
    for name in dir_names:
        file = open(f'arr_paths/CNB_arr_{name}.txt', 'r')
        for line in file.readlines():
            line = line.rstrip().split()
            arr_paths.append(line)
    return arr_paths


def read_dep_paths():
    dep_paths = []
    # dep_paths = []
    dir_names = ['D1', 'D2', 'D3', 'D4']
    for name in dir_names:
        file = open(f'dep_paths/CNB_dep_{name}.txt', 'r')
        for line in file.readlines():
            line = line.rstrip().split()
            dep_paths.append(line)
    return dep_paths


def read_edges(fname):
    edges = []
    file = open(fname, 'r')
    for line in file.readlines():
        line = line.rstrip().split(', ')
        x1 = line[0][1:]
        x2 = line[1][:-2]
        edges.append(tuple([x1, x2]))
    return edges


def read_layout(fname):
    glayout = {}
    file = open(fname, 'r')
    for line in file.readlines():
        line = line.rstrip().split()
        node = line[0]
        x = int(line[1])
        y = int(line[2])
        glayout[node] = [x, y]
    return glayout


def time_to_min(t):
    """t: string in %H:%M:%S"""
    h, m, s = tuple(map(int, t.split(":")))
    minutes = h * 60 + m
    return minutes


def process_solution():
    file = open('CNB_sched.txt', 'r')
    ans = []
    frames = {}
    for line in file.readlines():
        line = line.rstrip().split(",")
        ans.append(line)
    ans.sort(key=lambda x: time_to_min(x[4]))
    for a in ans:
        t0 = time_to_min(a[4])
        t1 = time_to_min(a[5])
        attrs = set()
        arr_path = a[2].split('-')
        dep_path = a[3].split('-')
        for i in arr_path:
            attrs.add(i)
        for i in dep_path:
            attrs.add(i)
        for i in range(len(arr_path)-1):
            attrs.add((arr_path[i], arr_path[i+1]))
        for i in range(len(dep_path)-1):
            attrs.add((dep_path[i], dep_path[i+1]))
        for i in range(t0, t1+1):
            if i not in frames:
                frames[i] = attrs
            else:
                frames[i].union(attrs)
    return frames


def update_graph(G, attrs):
    if attrs == None:
        attrs = set()

    for node in G.nodes():
        if node in attrs:
            node["active"] = 1
        else:
            node["active"] = 0


fig, ax = plt.subplots()
plt.axis('equal')
frames = process_solution()
station = nx.Graph()


def update(t):
    attrs = frames[t] if t in frames else None
    update_graph(station, attrs)


def main():

    edge_list = read_edges('CNB.txt')
    station.add_edges_from(edge_list)
    arr_paths = read_arr_paths()
    dep_paths = read_dep_paths()
    glayout = read_layout('CNB_layout.txt')
    n_active = 0
    e_active = 0
    nx.set_node_attributes(station, n_active, "active")
    nx.set_edge_attributes(station, e_active, "active")
    # frames = process_solution()
    print(station.edges[('D1', 'a')])


if __name__ == "__main__":
    main()
