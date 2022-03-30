import networkx as nx
import matplotlib.pyplot as plt
# from itertools import product
from datetime import datetime, time, timedelta
from mip import Model, BINARY, maximize, xsum, CBC


colors = {"arr": "green", "pf": "gray", "mid": "blue", "dep": "red"}
time_format = "%H:%M:%S"
date_format = "%d:%m:%Y"
cb_dummy = 1000
cb_regular = 1
headway = 0
tmax = 1440
gd = {}


def optimize(pt_graph, patterns):
    model = Model(solver_name=CBC)

    pnodes = pt_graph.nodes()
    pedges = pt_graph.edges()
    # y = [model.add_var(var_type=BINARY) for pf in platforms]
    # xtp = [model.add_var(var_type=BINARY) for node in pt_graph.nodes()]
    xtp = {node: model.add_var(var_type=BINARY) for node in pnodes}
    wtp = {node: pt_graph.nodes[node]['pattern'] for node in pnodes}

    # Objective
    c1 = xsum(-20*val*(abs(wtp[key][6])) for key, val in xtp.items())
    c2 = xsum(1000*val for key, val in xtp.items())
    c3 = xsum(10*val*wtp[key][7] for key, val in xtp.items())
    model.objective = maximize(c1 + c2 + c3)
    # xsum(val for key, val in xtp.items()))
    # xsum(val*(1+(-abs(wtp[key][-1]))) for key, val in xtp.items()))

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
    sol = []
    for k in range(model.num_solutions):
        for key, val in xtp.items():
            if val.x > 0:
                print(f'{key}:{val.x} {pt_graph.nodes[key]["pattern"]}')
                sol.append(pt_graph.nodes[key]["pattern"])
    return sol
    # print(f'')
    # model += xsum()


def main():
    station, directs, platforms, mid_nodes, arr_paths, dep_paths = create_station()
    # train_info_set = []
    # pprint(train_info_set)
    # train_set = create_train_set(train_info_set)
    train_set = read_trains("trains.txt")
    # pprint(train_set)
    for train in train_set:
        print(train)
    print("Patterns")
    patterns = create_train_patterns(
        train_set, platforms, arr_paths, dep_paths)
    for key in patterns.keys():
        print(f"Patterns for Train {key}")
        pt = patterns[key]
        # for p in pt:
        #     print(p)
    pt_graph = create_pattern_incompat_graph(patterns)
    pt_graph_labels = dict()
    # print('Patterns in PIG')
    for train, t_patterns in patterns.items():
        for i, pattern in enumerate(t_patterns):
            pt_graph_labels[f'{train}_{i}'] = f'{train}_{i}'
            # print(pt_graph.nodes[f'{train}_{i}']['pattern'])
    # nx.draw_shell(pt_graph, with_labels=True, labels=pt_graph_labels)
    # plt.show()
    # print(pt_graph.nodes())
    # print(pt_graph.edges())
    sol = optimize(pt_graph, patterns)
    write_schedule(sol)


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
        for i in range(19)
    ]

    # station.add_nodes_from(arr_directs)
    station.add_nodes_from(platforms)
    station.add_nodes_from(mid_nodes)
    station.add_nodes_from(directs)
    # station.add_nodes_from(dep_directs)
    edge_list = read_edges('CNB.txt')
    station.add_edges_from(edge_list)
    arr_paths = read_arr_paths()
    dep_paths = read_dep_paths()
    glayout = read_layout('CNB_layout.txt')
    # for v, data in station.nodes.data():
    #     print(v, data)
    # print(station.nodes.data())
    show_station(station, glayout)
    return station, directs, platforms, mid_nodes, arr_paths, dep_paths


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


def create_train_set(train_info_set):
    train_set = []
    for train in train_info_set:
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
                for i in range(-train.max_shift, train.max_shift + 1, 2):
                    # print(f"Here {i}")
                    arr_t = t0 + timedelta(minutes=i)
                    dep_t = arr_t + timedelta(minutes=stoppage)
                    shift = i
                    pref_pf = 1 if pf in train.pf_prefs else 0
                    # wp1 = stoppage+i
                    pattern = (
                        train.name,
                        pf[0],
                        a_path,
                        d_path,
                        arr_t.strftime(time_format),
                        dep_t.strftime(time_format),
                        shift,
                        pref_pf
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
    # create_station()
    main()
