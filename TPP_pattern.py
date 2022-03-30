import enum
from typing import no_type_check
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime, time, timedelta
from mip import Model, BINARY, maximize, xsum, CBC

colors = {"arr": "green", "pf": "gray", "mid": "blue", "dep": "red"}
time_format = "%H:%M:%S"
cb_dummy = 1000
cb_regular = 1
headway = 0
tmax = 1440
gd = {}
# Optimize Model


def optimize(pt_graph, patterns):
    model = Model(solver_name=CBC)

    pnodes = pt_graph.nodes()
    pedges = pt_graph.edges()
    # y = [model.add_var(var_type=BINARY) for pf in platforms]
    # xtp = [model.add_var(var_type=BINARY) for node in pt_graph.nodes()]
    xtp = {node: model.add_var(var_type=BINARY) for node in pnodes}
    wtp = {node: pt_graph.nodes[node]['pattern'] for node in pnodes}

    # Objective
    c1 = xsum(-20*val*(abs(wtp[key][-1])) for key, val in xtp.items())
    c2 = xsum(1000*val for key, val in xtp.items())
    model.objective = maximize(c1 + c2)
    # model.objective = maximize(
    #     xsum(val*(1+(1-abs(wtp[key][-1]))) for key, val in xtp.items()))

    # Constraint One : For Each Train 1 pattern is selected
    for train, t_patterns in patterns.items():
        model += xsum(xtp[f'{train}_{i}']
                      for i, pattern in enumerate(t_patterns)) <= 1
        # model += xsum(xtp[f'{train}_{i}']
        #               for i, pattern in enumerate(t_patterns)) > 0

    # Constraint Two : For Each pair of incompatible trains only select 1
    for u, v in pedges:
        utrain = u.split('_')[0]
        vtrain = v.split('_')[0]
        if utrain != vtrain:
            model += xtp[u] + xtp[v] <= 1

    model.optimize()

    print(f"Found {model.num_solutions} Solutions")
    for k in range(model.num_solutions):
        for key, val in xtp.items():
            if val.x > 0:
                print(f'{key}:{val.x} {pt_graph.nodes[key]["pattern"]}')
        # print(f'')
    # model += xsum()


# Main Function

def main():
    station = nx.DiGraph()
    arr_directs = [
        (f"A{i}", {"layer": 0, "ntype": "arr", "name": f"A{i}"}) for i in range(1, 3)
    ]
    for i in arr_directs:
        gd[i[0]] = 1
    dep_directs = [
        (f"D{i}", {"layer": 3, "ntype": "dep", "name": f"D{i}"}) for i in range(1, 3)
    ]
    for i in dep_directs:
        gd[i[0]] = 1
    platforms = [
        (f"P{i}", {"layer": 1, "ntype": "pf", "name": f"P{i}"}) for i in range(1, 4)
    ]
    mid_nodes = [
        (f"{chr(97 + i)}", {"layer": 2, "ntype": "mid", "name": f"{chr(97 + i)}"})
        for i in range(5)
    ]

    station.add_nodes_from(arr_directs)
    station.add_nodes_from(platforms)
    station.add_nodes_from(mid_nodes)
    station.add_nodes_from(dep_directs)

    edge_list = [
        ("A1", "a"),
        ("A1", "e"),
        ("A2", "d"),
        ("c", "D2"),
        ("b", "D1"),
        ("a", "P1"),
        ("a", "P2"),
        ("e", "P2"),
        ("P2", "b"),
        ("P3", "b"),
        ("P1", "c"),
        ("P2", "c"),
        ("d", "P2"),
        ("d", "P3"),
    ]
    station.add_edges_from(edge_list)

    arr_paths = arrival_paths(station, arr_directs, platforms)
    dep_paths = departure_paths(station, dep_directs, platforms)

    glayout = {
        "A1": [-10, 10],
        "D1": [-10, -10],
        "D2": [30, 10],
        "A2": [30, -10],
        "a": [0, 10],
        "b": [0, -10],
        "c": [20, 10],
        "d": [20, -10],
        "e": [0, 0],
        "P1": [10, 10],
        "P2": [10, 0],
        "P3": [10, -10],
    }

    print(arr_paths)
    print(dep_paths)
    # print(arr_paths)
    # a = ["a", "b", "c"]
    # b = [1, 2, 3]
    # c = product(a, b)
    # print(list(c))
    show_station(station, glayout)
    train_info_set = []

    create_train_info_set(
        train_info_set,
        "1",
        datetime.today(),
        datetime.strptime("09:00:00", time_format),
        datetime.strptime("09:03:00", time_format),
        10,
        3,
        "A1",
        "D2",
        ["P1", "P2"],
    )
    create_train_info_set(
        train_info_set,
        "2",
        datetime.today(),
        datetime.strptime("09:05:00", time_format),
        datetime.strptime("09:10:00", time_format),
        10,
        5,
        "A2",
        "D1",
        ["P2", "P3"],
    )
    create_train_info_set(
        train_info_set,
        "3",
        datetime.today(),
        datetime.strptime("09:09:00", time_format),
        datetime.strptime("09:13:00", time_format),
        10,
        4,
        "A2",
        "D2",
        ["P2"],
    )
    create_train_info_set(
        train_info_set,
        "4",
        datetime.today(),
        datetime.strptime("09:13:00", time_format),
        datetime.strptime("09:15:00", time_format),
        3,
        2,
        "A1",
        "D2",
        ["P1", "P2", "P3"],
    )
    create_train_info_set(
        train_info_set,
        "5",
        datetime.today(),
        datetime.strptime("09:13:00", time_format),
        datetime.strptime("09:16:00", time_format),
        4,
        3,
        "A1",
        "D1",
        ["P1", "P2", "P3"],
    )
    create_train_info_set(
        train_info_set,
        "6",
        datetime.today(),
        datetime.strptime("09:12:00", time_format),
        datetime.strptime("09:14:00", time_format),
        5,
        2,
        "A1",
        "D1",
        ["P1", "P3"],
    )
    create_train_info_set(
        train_info_set,
        "7",
        datetime.today(),
        datetime.strptime("09:00:00", time_format),
        datetime.strptime("09:05:00", time_format),
        6,
        5,
        "A1",
        "D1",
        ["P1", "P3"],
    )
    create_train_info_set(
        train_info_set,
        "8",
        datetime.today(),
        datetime.strptime("09:20:00", time_format),
        datetime.strptime("09:25:00", time_format),
        6,
        5,
        "A2",
        "D2",
        ["P1", "P2", "P3"],
    )
    # pprint(train_info_set)
    train_set = create_train_set(train_info_set)
    # pprint(train_set)
    for train in train_set:
        print(train)
    print("Patterns")
    patterns = create_train_patterns(
        train_set, platforms, arr_paths, dep_paths)
    for key in patterns.keys():
        print(f"Patterns for Train {key}")
        pt = patterns[key]
        for p in pt:
            print(p)
    pt_graph = create_pattern_incompat_graph(patterns)
    pt_graph_labels = dict()
    # print('Patterns in PIG')
    for train, t_patterns in patterns.items():
        for i, pattern in enumerate(t_patterns):
            pt_graph_labels[f'{train}_{i}'] = f'{train}_{i}'
            # print(pt_graph.nodes[f'{train}_{i}']['pattern'])
    nx.draw_shell(pt_graph, with_labels=True, labels=pt_graph_labels)
    plt.show()
    print(pt_graph.nodes())
    print(pt_graph.edges())
    optimize(pt_graph, patterns)


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


def arrival_paths(station, arr_directs, platforms):
    """returns list of arrival paths i.e. paths from direction
    node to platform node
    arr_directs: list of arr_direction attr
    """
    arr_path_names = []
    arr_paths = []
    for arr in arr_directs:
        for pf in platforms:
            arr_path_names.append((arr[0], pf[0]))
    for src in arr_path_names:
        for path in nx.all_simple_paths(station, source=src[0], target=src[1]):
            arr_paths.append(path)
    return arr_paths


def departure_paths(station, dep_directs, platforms):
    """returns list of departure paths i.e. paths from platform
    node to direction node
    arr_directs: list of dep_direction attr
    """
    dep_path_names = []
    dep_paths = []
    for dep in dep_directs:
        for pf in platforms:
            dep_path_names.append((pf[0], dep[0]))
    for src in dep_path_names:
        for path in nx.all_simple_paths(station, source=src[0], target=src[1]):
            dep_paths.append(path)
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


def create_train_set(train_list):
    train_set = []
    for train in train_list:
        train_set.append(Train(*train))
    return train_set


def create_train_info_set(train_info_set, *tinfo):
    info = tuple(tinfo)
    # print(info)
    train_info_set.append(info)
    return train_info_set


# def create_pattern(train, pf, arr_path, dep_path, arr_t, dep_t):
#     return tuple(train, pf, arr_path, dep_path, arr_t, dep_t)


def get_arr_paths(arr_paths, direct, pf):
    valid_paths = []
    for path in arr_paths:
        if path[0] == direct and path[-1] == pf[0]:
            valid_paths.append(path)
    # print(f"Valid Paths \n {valid_paths}")
    return valid_paths


def get_dep_paths(dep_paths, direct, pf):
    valid_paths = []
    for path in dep_paths:
        # print(f"{path} {direct} {pf}")
        if path[-1] == direct and path[0] == pf[0]:
            valid_paths.append(path)
    # print(f"Valid Paths \n {valid_paths}")
    return valid_paths


def train_pattern(train, platforms, arr_paths, dep_paths):
    arr_dir = train.arr_dir
    dep_dir = train.dep_dir
    patterns = []
    # print("Here")
    for pf in platforms:
        valid_arr_paths = get_arr_paths(arr_paths, arr_dir, pf)
        valid_dep_paths = get_dep_paths(dep_paths, dep_dir, pf)
        # print(valid_arr_paths)
        # print(valid_dep_paths)
        for a_path in valid_arr_paths:
            for d_path in valid_dep_paths:
                t0 = train.arr_t
                stoppage = train.min_stop
                # print(train.max_shift, list(range(1, train.max_shift + 1)))
                for i in range(-train.max_shift, train.max_shift + 1):
                    # print(f"Here {i}")
                    arr_t = t0 + timedelta(minutes=i)
                    dep_t = arr_t + timedelta(minutes=stoppage)
                    shift = i
                    # wp1 = stoppage+i
                    pattern = (
                        train.name,
                        pf[0],
                        a_path,
                        d_path,
                        arr_t.strftime(time_format),
                        dep_t.strftime(time_format),
                        shift
                    )
                    # print(pattern)
                    patterns.append(pattern)
    return patterns


def create_train_patterns(train_set, platforms, arr_paths, dep_paths):
    patterns = dict()
    for train in train_set:
        all_patterns = train_pattern(train, platforms, arr_paths, dep_paths)
        patterns[train.name] = all_patterns
    return patterns


def add_time(t1, dt):
    ans = t1 + timedelta(minutes=dt)
    return ans
    # print(f"Arr_t = {t1} \nUpdated = {ans}")


def check_incompat(p1, p2):
    if p1[1] == 'P0' or p2[1] == 'P0':
        return False
    if p1[0] == p2[0]:
        return False
    p1_t0 = (time_to_min(p1[4]))  # - headway) % tmax
    p1_t1 = (time_to_min(p1[5]))  # + headway + 1) % tmax
    p2_t0 = (time_to_min(p2[4]))  # - headway) % tmax
    p2_t1 = (time_to_min(p2[5]))  # + headway + 1) % tmax
    if not check_overlap(p1_t0, p1_t1 + 1, p2_t0, p2_t1 + 1):
        return False
    if p1[1] == p2[1]:
        return True
    p1_arr = p1[2]
    p2_arr = p2[2]
    p1_dep = p1[3]
    p2_dep = p2[3]

    for i in p1_arr:
        if i in p2_arr:
            return True
        elif i in p2_dep:
            return True

    for i in p1_dep:
        if i in p2_dep:
            return True
        elif i in p2_dep:
            return True

    return False


def time_to_min(t):
    """t: string in %H:%M:%S"""
    h, m, s = tuple(map(int, t.split(":")))
    minutes = h * 60 + m
    return minutes


def check_overlap(t0, t1, t2, t3):
    overlap = range(max(t0, t2), min(t1, t3))
    if len(overlap) == 0:
        return False
    return True


def create_pattern_incompat_graph(patterns):
    pt_graph = nx.Graph()
    nodes = []
    for train, all_pattern in patterns.items():
        for i, t_pattern in enumerate(all_pattern):
            pt_graph.add_node(f"{train}_{i}", pattern=t_pattern)
            nodes.append(f"{train}_{i}")
    for i, nodei in enumerate(nodes[:-1]):
        for j, nodej in enumerate(nodes[i+1:]):
            p1 = pt_graph.nodes[nodei]['pattern']
            p2 = pt_graph.nodes[nodej]['pattern']
            if check_incompat(p1, p2):
                pt_graph.add_edge(nodei, nodej)

    return pt_graph


if __name__ == "__main__":
    main()
    # print(time_to_min("23:59:00"))
    # time_format = "%H:%M:%S"
    # train_info_set = []
    # create_train_info_set(
    #     train_info_set,
    #     "1",
    #     datetime.today(),
    #     datetime.strptime("09:00:00", time_format),
    #     datetime.strptime("09:03:00", time_format),
    #     1,
    #     1,
    #     "A1",
    #     "D2",
    #     ["P1", "P2"],
    # )
    # train_set = create_train_set(train_info_set)
    # # pprint(train_set)
    # for train in train_set:
    #     print(train)
    # train = train_set[0]
    # print(train.arr_t)
    # print(type(train.arr_t))
    # add_time(train.arr_t, 5)
