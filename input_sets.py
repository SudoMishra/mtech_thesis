from datetime import date
from datetime import datetime
from datetime import timedelta
from pprint import pprint
from collections import defaultdict


class node:
    """
    Class for all nodes used in the station topology.

    Attributes:
        name (str): The name of the node.
        node_type (str): Type of the node, can be {pf, dir, mid}.
        color (str): color of the node.
        travel_time: The travel time for each direction (node_type = dir). None for other types of nodes.
    """

    colors = {
        "pf": "Grey",
        "dir": "Green",
        "mid": "Blue",
    }

    def __init__(self, name, node_type):
        self.name = str(name)
        self.node_type = str(node_type)
        self._set_color(node_type)
        self._set_travel_time()

    def __repr__(self) -> str:
        return f"\n Node Type: {self.node_type} \n Name: {self.name}"

    def __str__(self) -> str:
        return f" Node Type: {self.node_type} \n Name: {self.name}"

    def _set_color(self, node_type):
        """ Sets the color of the node."""
        self.color = node.colors[node_type]

    def _set_travel_time(self, t=1):
        """ Sets the travel time for direction nodes, None otherwise."""
        if self.node_type == "dir":
            self.travel_time = t
        else:
            self.travel_time = None


class Path:
    """
    Class for all paths used in the station topology.

    Attributes:
        path (list): List of all edges in the path.
        nodes (list): List of all nodes in the path.
        ptype (str): arrival if path starts from direction, departure if path starts from platform.
    """
    ptypes = {
        "dirpf": "arrival",
        "pfdir": "departure"
    }

    def __init__(self, nodes):
        self.path = []
        self.nodes = nodes
        self.direction = nodes[0].name if nodes[0].node_type == "dir" else nodes[-1].name
        for i in range(len(nodes)-1):
            self.path.append((nodes[i], nodes[i+1]))
        self.ptype = Path.ptypes[nodes[0].node_type+nodes[-1].node_type]

    def __repr__(self) -> str:
        return f"\n Path Type: {self.ptype} \n Nodes: {[node.name for node in self.nodes]}"

    def __str__(self) -> str:
        return f"Path Type: {self.ptype} \n Nodes: {[node.name for node in self.nodes]}"


# Platform Set


def create_platform_nodes(pfs):
    """ Given a list of platform names return a list of all platforms. """
    pf_nodes = []
    for i in pfs:
        pf_nodes.append(node(i, "pf"))
    return pf_nodes


# # A dictionary to store all platforms where key is the platform name(str)
# B = dict()
# pf_nodes = create_platform_nodes([1, 2, 3])
# for pf in pf_nodes:
#     B[pf.name] = pf

# Direction Set


def create_direction_nodes(dirs_):
    """ Given a list of direction names return a list of all directions. """
    dir_nodes = []
    for i in dirs_:
        dir_nodes.append(node(i, "dir"))
    return dir_nodes


# # A dictionary to store all directions where key is the direction name(str)
# D = dict()

# dir_nodes = create_direction_nodes([1, 2])
# for dir_ in dir_nodes:
#     D[dir_.name] = dir_

# Mid node set


def create_mid_nodes(mid):
    """ Given a list of middle node names return a list of all middle nodes. """
    mid_nodes = []
    for i in mid:
        mid_nodes.append(node(i, "mid"))
    return mid_nodes


# # A dictionary to store all middle nodes where key is the middle node name(str)
# M = dict()

# mid_nodes = create_mid_nodes(['a', 'b', 'c', 'd', 'e'])
# for mid in mid_nodes:
#     M[mid.name] = mid


# Path Set

# # A dictionary to store all paths where key is the direction name(str)
# R = dict()
# Structure R -> arr -> dirs -> [paths] or dep -> pfs -> [paths]

def create_path_set(n_dir):

    R = dict()
    R['arr'] = defaultdict(list)
    R['dep'] = defaultdict(list)
    for i in range(1, n_dir+1):
        R['arr'][str(i)] = []
        R['dep'][str(i)] = []

    return R

# R[1] = Path([D["1"], M["a"], B["1"]])
# R[2] = Path([D["1"], M["a"], B["2"]])
# R[3] = Path([D["1"], M["e"], B["2"]])
# R[4] = Path([D["2"], M["d"], B["2"]])
# R[5] = Path([D["2"], M["d"], B["3"]])
# R[6] = Path([B["2"], M["b"], D["1"]])
# R[7] = Path([B["3"], M["b"], D["1"]])
# R[8] = Path([B["1"], M["c"], D["2"]])
# R[9] = Path([B["2"], M["c"], D["2"]])

# Incompatible Path Set


def path_compat(path1, path2):
    # check nodes intersection
    # flag = True
    l1 = len(path1.nodes)
    l2 = len(path2.nodes)

    # Checking common middle nodes
    for i, node in enumerate(path1.nodes[1:-1]):
        if node in path2.nodes:
            return False

    # Optional: Checking common edges
    for i, edge in enumerate(path1.path):
        if edge in path2.path:
            return False

    return True


def incompat_path_list(R, path):
    incompat_set = []
    for i in R.keys():
        if not path_compat(path, R[i]):
            incompat_set.append(R[i])
    return incompat_set


def create_incompat_path_set(R):
    I = dict()
    for i in R.keys():
        I[i] = incompat_path_list(R, R[i])
    return I


# Train class

class Train:
    def __init__(self, *args):
        self.name = args[0]
        self.date = args[1].date()
        # datetime.combine(self.date, args[2].time()).time()
        self.arr_t = args[2].time()
        # datetime.combine(self.date, args[3].time()).time()
        self.dep_t = args[3].time()
        self.max_shift = args[4]
        self.min_stop = args[5]
        self.arr_dir = args[6]
        self.dep_dir = args[7]
        self.ct = args[8]

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

# Train Set


# train_info_set = set()


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

# Preference list of platforms for each pair of directions
# currently none


# Pattern Set for train t
def create_pattern(pf, arr_path, dep_path, arr_time, dep_time):
    return tuple(pf, arr_path, dep_path, arr_time, dep_time)


def get_minute_time_diff(t1, t2):
    return (t2-t1).total_seconds()/60


def check_pattern_compatibility(train, pattern):
    pf, arr_path, dep_path, arr_time, dep_time = pattern
    # if t_name != train.name:
    #     return False
    if pf not in train.ct:
        return False
    if arr_path.nodes[0].name != train.arr_dir:
        return False
    if dep_path.nodes[0].name != train.dep_dir:
        return False
    if abs(get_minute_time_diff(train.arr_t, arr_time)) > train.max_shift:
        return False
    if abs(get_minute_time_diff(train.dep_t, dep_time)) > train.max_shift:
        return False

    return True


def get_arr_paths(R, d, pf):
    paths = []
    for path in R['arr'][str(d)]:
        if path.nodes[-1].name == pf:
            paths.append(path)
    return paths


def get_dep_paths(R, pf, d):
    paths = []
    for path in R['dep'][str(pf)]:
        if path.nodes[-1].name == str(d):
            paths.append(path)
    return paths


def get_arr_times(arr_time, shift):
    ans = []
    for dt in range(shift+1):
        t = arr_time + timedelta(minutes=dt)
        ans.append(t)
        t = arr_time - timedelta(minutes=dt)
        ans.append(t)
    return ans


def get_dep_times(arr_time, dep_time, stop, shift):
    ans = []
    for dt in range(shift+1):
        t = dep_time + timedelta(minutes=dt)
        if get_minute_time_diff(t, arr_time) < stop:
            continue
        ans.append(t)
        t = dep_time - timedelta(minutes=dt)
        if get_minute_time_diff(t, arr_time) < stop:
            continue
        ans.append(t)
    return ans


def create_train_pattern(R, train):
    Pt = set()
    for pf in train.ct:
        arr_paths = get_arr_paths(R, train.arr_dir, pf)
        dep_paths = get_dep_paths(R, pf, train.dep_dir)
        for arr_path in arr_paths:
            for dep_path in dep_paths:
                arr_times = get_arr_times(train.arr_t, train.max_shift)
                for arr_time in arr_times:
                    dep_times = get_dep_times(
                        arr_time, train.dep_time, train.stop, train.max_shift)
                    for dep_time in dep_times:
                        pattern = create_pattern(
                            pf, arr_path, dep_path, arr_time, dep_time)
                        Pt.add(pattern)


def create_all_train_pattern_set(R, train_set):
    P = dict()
    for train in train_set:
        P[train.name] = create_train_pattern(R, train)
    return P


def create_pattern_incompat_graph(P):
    pass


def main():
    """ Main function to run program """
    # A dictionary to store all platforms where key is the platform name(str)
    B = dict()
    pf_nodes = create_platform_nodes([1, 2, 3])
    for pf in pf_nodes:
        B[pf.name] = pf

    pprint("Platform Nodes")
    pprint(B)

    # A dictionary to store all directions where key is the direction name(str)
    D = dict()

    dir_nodes = create_direction_nodes(["1", "2"])
    for dir_ in dir_nodes:
        D[dir_.name] = dir_

    pprint("Dir Nodes")
    pprint(D)

    # A dictionary to store all middle nodes where key is the middle node name(str)
    M = dict()

    mid_nodes = create_mid_nodes(['a', 'b', 'c', 'd', 'e'])
    for mid in mid_nodes:
        M[mid.name] = mid

    pprint("Mid Nodes")
    pprint(M)

    # A dictionary to store all paths where key is the direction name(str)
    R = create_path_set(2)

    # pprint(R)

    R['arr']['1'].append(Path([D["1"], M["a"], B["1"]]))
    R['arr']['1'].append(Path([D["1"], M["a"], B["2"]]))
    R['arr']['1'].append(Path([D["1"], M["e"], B["2"]]))
    R['arr']['2'].append(Path([D["2"], M["d"], B["2"]]))
    R['arr']['2'].append(Path([D["2"], M["d"], B["3"]]))
    R['dep']['2'].append(Path([B["2"], M["b"], D["1"]]))
    R['dep']['3'].append(Path([B["3"], M["b"], D["1"]]))
    R['dep']['1'].append(Path([B["1"], M["c"], D["2"]]))
    R['dep']['2'].append(Path([B["2"], M["c"], D["2"]]))

    pprint("Path Set")
    pprint(R)
    # Train Set

    train_info_set = []
    time_format = '%H:%M:%S'
    create_train_info_set(train_info_set,
                          "1",
                          datetime.today(),
                          datetime.strptime('09:00:00', time_format),
                          datetime.strptime('09:03:00', time_format),
                          1, 1, 'D1', 'D2', [1, 2])
    create_train_info_set(train_info_set,
                          "2",
                          datetime.today(),
                          datetime.strptime('09:05:00', time_format),
                          datetime.strptime('09:10:00', time_format),
                          1, 1, 'D2', 'D1', [2, 3])
    create_train_info_set(train_info_set,
                          "3",
                          datetime.today(),
                          datetime.strptime('09:09:00', time_format),
                          datetime.strptime('09:12:00', time_format),
                          1, 1, 'D1', 'D1', [2])
    # pprint(train_info_set)
    train_set = create_train_set(train_info_set)
    # pprint(train_set)
    for train in train_set:
        print(train)


if __name__ == "__main__":
    main()
