import networkx as nx
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt

colors = {"arr": "green", "pf": "gray", "mid": "blue", "dep": "red"}
time_format = "%H:%M:%S"
date_format = "%d:%m:%Y"
cb_dummy = 1000
cb_regular = 1
headway = 0
tmax = 1440
gd = {}


def write_schedule(sol):
    file = open('CNB_sched.txt', 'w')
    for line in sol:
        tname, pf, arr_path, dep_path, arr_t, dep_t, shift, _ = line
        ans = f'{tname},{pf},{"-".join(arr_path)},{"-".join(dep_path)},{arr_t},{dep_t},{shift}\n'
        file.writelines(ans)
    file.close()


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


def read_trains(fname):
    train_info_set = []
    file = open(fname, 'r')
    for line in file.readlines():
        line = line.rstrip().split()
        name = line[0]
        curr_date = datetime.strptime(line[1], date_format)
        arr_t = datetime.strptime(line[2], time_format)
        dep_t = datetime.strptime(line[3], time_format)
        max_shift = int(line[4])
        stop = int(line[5])
        arr_dir = line[6]
        dep_dir = line[7]
        pref_pf = line[8].split('-')
        data = (name, curr_date, arr_t, dep_t,
                max_shift, stop, arr_dir, dep_dir, pref_pf)
        train_info_set.append(Train(*data))
    return train_info_set


def create_station():
    station = nx.Graph()
    directs = [
        (f"D{i}", {"ntype": "arr", "name": f"D{i}"}) for i in range(1, 5)
    ]
    platforms = [
        (f"P{i}", {"ntype": "pf", "name": f"P{i}"}) for i in range(1, 11)
    ]
    mid_nodes = [
        (f"{chr(97 + i)}", {"ntype": "mid", "name": f"{chr(97 + i)}"})
        for i in range(5)
    ]

    # station.add_nodes_from(arr_directs)
    station.add_nodes_from(platforms)
    station.add_nodes_from(mid_nodes)
    station.add_nodes_from(directs)
    # station.add_nodes_from(dep_directs)
    edge_list = read_edges('test_station.txt')
    station.add_edges_from(edge_list)
    # arr_paths = read_arr_paths()
    # dep_paths = read_dep_paths()
    glayout = read_layout('test_layout.txt')
    # for v, data in station.nodes.data():
    #     print(v, data)
    # print(station.nodes.data())
    show_station(station, glayout)
    return station, directs, platforms, mid_nodes  # , arr_paths, dep_paths


def show_station(G, glayout):
    """Function shows graph in matplotlib
    G: Graph
    glayout: The coordinates of the nodes in the graph to show
    """

    color = [colors[data["ntype"]] for v, data in G.nodes.data()]
    # for node, data in G.nodes.data():
    #     print(node, data)
    label_dict = {node: data["name"] for node, data in G.nodes.data()}
    # pos = nx.multipartite_layout(G, subset_key="layer")
    # print(pos)
    plt.figure(figsize=(8, 8))
    nx.draw(G, glayout, node_color=color, with_labels=True, labels=label_dict)
    plt.axis("equal")
    plt.show()


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


class Train:
    def __init__(self, *args):
        self.name = args[0]
        self.date = args[1].date()
        self.arr_t = datetime.combine(self.date, args[2].time())  # .time()
        # self.arr_t = args[2].time()
        self.dep_t = datetime.combine(self.date, args[3].time())  # .time()
        # self.dep_t = args[3].time()
        self.max_shift = args[4]
        self.min_stop = args[5]
        self.arr_dir = args[6]
        self.dep_dir = args[7]
        self.pf_prefs = args[8]

    def possible_arr_t(self, t):
        if (t >= self.arr_t - self.max_shift) and (t <= self.arr_t + self.max_shift):
            return True
        else:
            return False

    def possible_dep_t(self, t):
        if (t >= self.dep_t - self.max_shift) and (t <= self.dep_t + self.max_shift):
            return True
        else:
            return False

    def __repr__(self) -> str:
        return f"\n Name: {self.name} \n Date: {self.date} \n Arrival Time: {self.arr_t} \n Departure Time: {self.dep_t} \n Arr/Dep Dir: {self.arr_dir,self.dep_dir}"

    def __str__(self) -> str:
        return f"Name: {self.name} \n Date: {self.date} \n Arrival Time: {self.arr_t} \n Departure Time: {self.dep_t} \n Arr/Dep Dir: {self.arr_dir,self.dep_dir}"


class Env:
    def __init__(self):
        pass


if __name__ == "__main__":
    create_station()
