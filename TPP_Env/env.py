import networkx as nx
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import heapq
import random
import dill

colors = {"arr": "green", "pf": "gray", "mid": "blue", "dep": "red"}
time_format = "%H:%M:%S"
date_format = "%d:%m:%Y"
cb_dummy = 1000
cb_regular = 1
headway = 0
tmax = 1440
gd = {}
n_pfs = 3
n_dir = 2
n_mid = 5
n_prty = 3
# seed = 42
# random.seed(seed)
# In_paths with their ids : [Path]
arr_paths = {}
# Out Paths with their ids : [Path]
dep_paths = {}
# pf id
pf_id = {}
# in Paths from a given direction (dir : [in_path ids])
in_paths_from = {}
# out Paths from a given platform to a given direction (pf,dir: [out paths ids])
out_paths_from = {}
# pig for
PIG = nx.Graph()
PIG_labels = dict()


def check_incompat(p1, p2):
    for d in p1:
        if d[0] == "D":
            continue
        if d in p2:
            return True
    return False


def get_available(paths):
    avlbl = []
    for idx, path in enumerate(paths):
        if not path:
            avlbl.append(idx)
    return avlbl


def create_PIG_nodes():
    for in_dir, in_paths in in_paths_from.items():
        for in_path_id in in_paths:
            # print(f"in_path_id - {in_path_id} : {arr_paths[in_path_id][-1]}")
            pf = arr_paths[in_path_id][-1]
            PIG.add_node(f"{in_dir}-{in_path_id}", path_type=0,
                         data=arr_paths[in_path_id], idx=in_path_id, pf=pf)
            PIG_labels[f"{in_dir}-{in_path_id}"] = arr_paths[in_path_id]
    for pf, out_dir_dict in out_paths_from.items():
        for out_dir, out_paths in out_dir_dict.items():
            for out_path_id in out_paths:
                pf = dep_paths[out_path_id][0]
                PIG.add_node(f"{out_dir}-{out_path_id}",
                             path_type=1, data=dep_paths[out_path_id], idx=out_path_id, pf=pf)
                PIG_labels[f"{in_dir}-{in_path_id}"] = dep_paths[out_path_id]


def create_PIG_edges():
    # in_path_edges
    labels = {
        0: "in-in",
        1: "in-out",
        2: "out-out"
    }
    for idx, n1 in enumerate(list(PIG.nodes)[:-1]):
        for idx2, n2 in enumerate(list(PIG.nodes)[idx+1:]):
            node1 = PIG.nodes[n1]
            node2 = PIG.nodes[n2]
            if check_incompat(node1["data"], node2["data"]):
                edge_type = node1["path_type"] + node2["path_type"]
                PIG.add_edge(n1, n2, edge_type=labels[edge_type])
                # PIG_labels[f"{n1}-{n2}"] = labels[edge_type]


def get_in_paths_from(arr_paths):
    for idx, path in enumerate(arr_paths):
        if path[0] not in in_paths_from.keys():
            in_paths_from[path[0]] = []
        in_paths_from[path[0]].append(idx)


def get_out_paths_from(dep_paths):
    for idx, path in enumerate(dep_paths):
        if path[0] not in out_paths_from.keys():
            out_paths_from[path[0]] = {}
        if path[-1] not in out_paths_from[path[0]].keys():
            out_paths_from[path[0]][path[-1]] = []
        out_paths_from[path[0]][path[-1]].append(idx)


def get_pf_ids(pf):
    for idx, p in enumerate(pf):
        pf_id[p[0]] = idx


def one_hot(data, n):
    val = [1 if i in data else 0 for i in range(n)]
    return val


class PriorityQueue:
    def __init__(self):
        self.q = []

    def __len__(self):
        return len(self.q)

    def __iter__(self):
        for i, _ in enumerate(self.q):
            yield self.q[i]

    def hsort(self):
        heapq.heapify(self.q)

    def push(self, data):
        heapq.heappush(self.q, data)

    def pop(self):
        return heapq.heappop(self.q)


class Msg:
    def __init__(self, res, rel_time, nbr_node):
        '''
        Message to be stored in the environment to release locked path in the future
        res : (in_path id, out_path id, pf id, stoppage time)
        rel_time : time at which to release the resource
        nbr_node : 1 if resource is only single path else 0
        '''
        self.res = res
        self.rel_time = rel_time
        self.nbr_node = nbr_node

    def __lt__(self, other):
        if self.rel_time < other.rel_time:
            return True
        else:
            return False


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
        # print([x1, x2])
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
    '''
        data is a dict having
        name    : name of Train
        in_dir  : incoming direction (0 ... n_dirs)
        out_dir : outgoing direction (0 ... n_dirs)
        arr_t   : arrival time (0 .. 1440)
        stop    : stoppage time
        pref_pf : list of preferred platforms
        prty    : priority (1 .. 4) higher has more priority
    '''
    train_info_set = []
    file = open(fname, 'r')
    for line in file.readlines():
        line = line.rstrip().split()
        name = line[0]
        curr_date = datetime.strptime(line[1], date_format)
        in_dir = line[2]
        out_dir = line[3]
        arr_t = int(line[4])
        stop = int(line[5])
        pref_pf = line[6].split('-')
        prty = line[7]
        data = {
            "name": name,
            "in_dir": in_dir,
            "out_dir": out_dir,
            "arr_t": arr_t,
            "stop": stop,
            "pref_pf": pref_pf,
            "prty": prty
        }
        train_info_set.append(Train(data))
    return train_info_set


def create_station():
    station = nx.Graph()
    directs = [
        (f"D{i}", {"ntype": "arr", "name": f"D{i}"}) for i in range(1, n_dir+1)
    ]
    platforms = [
        (f"P{i}", {"ntype": "pf", "name": f"P{i}"}) for i in range(1, n_pfs+1)
    ]
    mid_nodes = [
        (f"{chr(97 + i)}", {"ntype": "mid", "name": f"{chr(97 + i)}"})
        for i in range(n_mid)
    ]

    # station.add_nodes_from(arr_directs)
    station.add_nodes_from(platforms)
    station.add_nodes_from(mid_nodes)
    station.add_nodes_from(directs)
    # station.add_nodes_from(dep_directs)
    edge_list = read_edges('test_station.txt')
    station.add_edges_from(edge_list)
    arr_paths = read_arr_paths()
    dep_paths = read_dep_paths()
    glayout = read_layout('test_layout.txt')
    # for v, data in station.nodes.data():
    #     print(v, data)
    # print(station.nodes.data())
    # show_station(station, glayout)
    return station, directs, platforms, mid_nodes, arr_paths, dep_paths


def show_station(G, glayout):
    """Function shows graph in matplotlib
    G: Graph
    glayout: The coordinates of the nodes in the graph to show
    """
    # print(G.nodes.data())
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
    dir_names = [f"D{i}" for i in range(1, n_dir+1)]
    for name in dir_names:
        file = open(f'arr_paths/test_arr_{name}.txt', 'r')
        for line in file.readlines():
            line = line.rstrip().split()
            arr_paths.append(line)
    return arr_paths


def read_dep_paths():
    dep_paths = []
    # dep_paths = []
    dir_names = [f"D{i}" for i in range(1, n_dir+1)]
    for name in dir_names:
        file = open(f'dep_paths/test_dep_{name}.txt', 'r')
        for line in file.readlines():
            line = line.rstrip().split()
            dep_paths.append(line)
    return dep_paths


def save_agent(agent, fname):
    dill.dump(agent, file=open(f'./saved_agents/{fname}.pickle', 'wb'))


def load_agent(fname):
    with open(f"./saved_agents/{fname}.pickle", "rb") as file:
        agent = dill.load(file)
    return agent


class Train:
    def __init__(self, data):
        '''
        data is a dict having
        name    : name of Train
        in_dir  : incoming direction (0 ... n_dirs)
        out_dir : outgoing direction (0 ... n_dirs)
        arr_t   : arrival time (0 .. 1440)
        stop    : stoppage time
        pref_pf : list of preferred platforms
        prty    : priority (1 .. 4) higher has more priority
        '''
        self.data = dict()
        for key, value in data.items():
            self.data[key] = value
        self.reward = 0
        # self.in_dir = one_hot([self.data["in_dir"]],  n_dir)
        # self.out_dir = one_hot(self.data["out_dir"], n_dir)
        # self.pfs = one_hot(self.data["pref_pf"], n_pfs)
        # self.prty = one_hot([self.data["prty"]], n_prty)

    def __str__(self):
        name = f'Train Name: {self.data["name"]}\n'
        in_dir = f'In Direction: {self.data["in_dir"]}\n'
        out_dir = f'Out Direction: {self.data["out_dir"]}\n'
        arr_t = f'Arrival Time: {self.data["arr_t"]}\n'
        stop = f'Stoppage: {self.data["stop"]}\n'
        pref_pfs = f'Pref Pfs: {self.data["pref_pf"]}\n'
        prty = f'Priority: {self.data["prty"]}\n'

        return name + arr_t + prty

    def __lt__(self, other):
        if self.data["arr_t"] < other.data["arr_t"]:
            return True
        elif self.data["arr_t"] == other.data["arr_t"]:
            if self.data["prty"] > other.data["prty"]:
                return True
            else:
                return False
        else:
            return False

    def get_dep_time(self):
        return self.data["arr_t"] + self.data["stop"]

    def delay(self):
        self.data["arr_t"] = (self.data["arr_t"] + 1) % tmax


class Agent:
    def __init__(self):
        pass

    def act(self, train, data, tq):
        '''
        Get an action from an Agent
        action : (train,in_path,out_path,pf,delay)
        '''
        in_dir = train.data["in_dir"]
        out_dir = train.data["out_dir"]
        # get in_paths
        possible_in_paths = []
        for path_id in in_paths_from[in_dir]:
            if not data[0][path_id]:
                possible_in_paths.append(path_id)
        if not len(possible_in_paths):
            return (train, -1, -1, -1, 1)
        in_path = self.choose_in_path(possible_in_paths)
        pf = arr_paths[in_path][-1]
        possible_out_paths = []
        for path_id in out_paths_from[pf][out_dir]:
            if not data[1][path_id]:
                possible_out_paths.append(path_id)
        if not len(possible_out_paths):
            return (train, -1, -1, -1, 1)
        out_path = self.choose_out_path(possible_out_paths)
        return (train, in_path, out_path, pf, 0)

    def choose_in_path(self, in_paths):
        '''
        returns the id of the chosen in_path
        '''
        return random.choice(in_paths)

    def choose_out_path(self, out_paths):
        '''
        returns the id of the chosen in_path
        '''
        return random.choice(out_paths)

    def get_reward(self, reward):
        pass


class Agent_MCTS:
    def __init__(self, t_list, in_state, out_state):
        self.n_train = 0  # no. of episodes trained
        self.q_table = dict()
        for t in t_list:
            self.q_table[t.data["name"]] = {
                "in_path_val": [0 for i in range(len(in_state))],
                "out_path_val": [0 for i in range(len(out_state))],
                "in_path_count": [0 for i in range(len(in_state))],
                "out_path_count": [0 for i in range(len(out_state))]}

    def act(self, train, data, tq):
        '''
        Get an action from an Agent
        action : (train,in_path,out_path,pf,delay)
        '''
        in_dir = train.data["in_dir"]
        out_dir = train.data["out_dir"]
        # get in_paths
        possible_in_paths = []
        for path_id in in_paths_from[in_dir]:
            if not data[0][path_id]:  # if  in_path_id is not locked
                possible_in_paths.append(path_id)
        if not len(possible_in_paths):  # if there are no possible in_paths no choice but delay
            return (train, -1, -1, -1, 1)
        # choose the in_path
        in_path = self.choose_in_path(possible_in_paths)
        # get platform from chosen in_path
        pf = arr_paths[in_path][-1]
        possible_out_paths = []
        for path_id in out_paths_from[pf][out_dir]:
            if not data[1][path_id]:  # if out_path_id is not locked
                possible_out_paths.append(path_id)
        if not len(possible_out_paths):
            return (train, -1, -1, -1, 1)
        out_path = self.choose_out_path(possible_out_paths)
        return (train, in_path, out_path, pf, 0)

    def choose_in_path(self, in_paths):
        '''
        returns the id of the chosen in_path
        '''
        return random.choice(in_paths)

    def choose_out_path(self, out_paths):
        '''
        returns the id of the chosen in_path
        '''
        return random.choice(out_paths)

    def get_reward(self, reward):
        pass

    def update_q_table(self, rewards):
        '''
        rewards : list of (train_id, in_path_id, out_path_id, net returns)
        '''
        v1, v2, v3, v4 = 0, 0, 0, 0
        for r in rewards:
            self.q_table[r[0]]["in_path_count"][r[1]] += 1
            self.q_table[r[0]]["out_path_count"][r[2]] += 1
            v1 = self.q_table[r[0]]["in_path_val"][r[1]]
            v2 = self.q_table[r[0]]["out_path_val"][r[2]]
            v3 = self.q_table[r[0]]["in_path_count"][r[1]]
            v4 = self.q_table[r[0]]["out_path_count"][r[2]]
            self.q_table[r[0]]["in_path_val"][r[1]] = v1 + (r[3]-v1)/v3
            self.q_table[r[0]]["out_path_val"][r[1]] = v2 + (r[3]-v2)/v4


class Env:
    '''
    init_conds  : A dictionary with keys
                    "tlist"      : (list of trains),
                    "start_time" :(start time of clock)
                    "in_state"   : initial state of inbound paths
                    "out_state"  : initial state of outbound
                    "pf_state"   : initial state of platforms
    '''

    def __init__(self, init_conds):
        self.station, self.directs, self.platforms, self.mid_nodes, self.arr_paths, self.dep_paths = create_station()
        self.in_state = [0 for i in range(len(self.arr_paths))]
        # stores the time when in_path will be unlocked it is 0 if already unlocked
        self.in_state_t = [0 for i in range(len(self.arr_paths))]
        self.out_state = [0 for i in range(len(self.dep_paths))]
        # stores the time when out_path will be unlocked it is 0 if already unlocked
        self.out_state_t = [0 for i in range(len(self.dep_paths))]
        self.pf_state = [0 for i in range(n_pfs)]
        # stores the time when pf will be unlocked it is 0 if already unlocked
        self.pf_state_t = [0 for i in range(n_pfs)]
        self.tq = PriorityQueue()
        self.mq = PriorityQueue()
        self.clock = init_conds["start_time"]
        self.schedule = []
        self.agent = None
        self.ep_limit = 100
        self.net_delay = 0
        # self.t_list = dict() # (t_name,idx_when_scheduled,reward)
        self.ep_returns = []
        self.set_ids()
        get_in_paths_from(self.arr_paths)
        get_out_paths_from(self.dep_paths)
        get_pf_ids(self.platforms)
        create_PIG_nodes()
        create_PIG_edges()
        self.init_states(init_conds)
        self.init_pq(init_conds["tlist"], init_conds["mlist"])

    def set_ids(self):
        for idx, arr in enumerate(self.arr_paths):
            arr_paths[idx] = arr
        for idx, arr in enumerate(self.dep_paths):
            dep_paths[idx] = arr

    def init_agent(self, agent):
        self.agent = agent

    def init_pq(self, tlist, mlist):
        for t in tlist:
            self.tq.push(t)
        for m in mlist:
            self.mq.push(m)

    def init_states(self, init_conds):
        if len(init_conds["in_state"]):
            self.in_state = init_conds["in_state"]
        if len(init_conds["out_state"]):
            self.out_state = init_conds["out_state"]
        if len(init_conds["pf_state"]):
            self.pf_state = init_conds["pf_state"]
        self.in_state_t = [0 for i in range(len(self.arr_paths))]
        self.out_state_t = [0 for i in range(len(self.dep_paths))]
        self.pf_state_t = [0 for i in range(n_pfs)]

    def reset(self, init_conds):
        self.tq = PriorityQueue()
        self.mq = PriorityQueue()
        self.init_states(init_conds)
        self.init_pq(init_conds["tlist"], init_conds["mlist"])
        self.clock = init_conds["start_time"]
        self.net_delay = 0
        self.schedule = []
        self.ep_returns = []

    def lock(self, res):
        '''
        Locks a resource in the env
        res : (in_path id, out_path id, pf id, stoppage time)
        '''
        curr = self.clock
        in_path, out_path, pf, stop = res[0], res[1], res[2], res[3]
        pf = pf_id[pf]
        self.in_state[in_path] += 1
        self.in_state_t[in_path] = curr + stop
        self.out_state[out_path] += 1
        self.out_state_t[out_path] = curr + stop
        # if pf != -1:
        self.pf_state[pf] += 1
        self.pf_state_t[pf] = curr + stop
        # print(f"Locked nbr in_path {in_path} out_path {out_path}")
        print(f"Locked in_path {in_path} out_path {out_path} pf {pf}")
        curr = (curr + stop) % tmax
        msg = Msg(res, curr, 0)
        self.mq.push(msg)

    def lock_nbrs(self, res):
        '''
        Locks a particular in_path or out_path
        res : (in_path_id,out_path_id,pf,stoppage_time)
        pf = -1
        if id == -1 then dont lock
        '''
        curr = self.clock
        in_path, out_path, stop = res[0], res[1], res[3]
        # pf = pf_id[pf]
        if in_path != -1:
            self.in_state[in_path] += 1
            self.in_state_t[in_path] = curr + stop
        if out_path != -1:
            self.out_state[out_path] += 1
            self.out_state_t[out_path] = curr + stop
        # if pf != -1:
        # self.pf_state[pf] = 1
        print(f"Locked nbr in_path {in_path} out_path {out_path}")
        # print(f"Locked in_path {in_path} out_path {out_path} pf {pf}")
        curr = (curr + stop) % tmax
        msg = Msg(res, curr, 1)
        self.mq.push(msg)

    def unlock(self, msg):
        '''
        Unlocks a resource based on the msg in the env
        res : (in_path id, out_path id, pf id, stoppage time)
        '''
        in_path, out_path, pf = msg.res[0], msg.res[1], msg.res[2]
        # if pf != -1:
        pf = pf_id[pf]
        self.in_state[in_path] -= 1
        self.in_state_t[in_path] = 0
        self.out_state[out_path] -= 1
        self.out_state_t[out_path] = 0
        # if pf != -1:
        self.pf_state[pf] -= 1
        self.pf_state_t[pf] = 0
        # print(f"UnLocked nbr in_path {in_path} out_path {out_path}")
        print(f"UnLocked in_path {in_path} out_path {out_path} pf {pf}")

    def unlock_nbrs(self, msg):
        '''
        Unlocks a resource based on the msg in the env
        res : (in_path id, out_path id, pf id, stoppage time)
        '''
        in_path, out_path = msg.res[0], msg.res[1]
        # if pf != -1:
        #     pf = pf_id[pf]
        if in_path != -1:
            self.in_state[in_path] -= 1
            self.in_state_t[in_path] = 0
        if out_path != -1:
            self.out_state[out_path] -= 1
            self.out_state_t[out_path] = 0
        # if pf != -1:
        # self.pf_state[pf] = 0
        print(f"UnLocked nbr in_path {in_path} out_path {out_path}")
        # print(f"UnLocked in_path {in_path} out_path {out_path} pf {pf}")

    def check_msg(self):
        '''
        Checks the message queue to unlock trains
        '''
        if len(self.mq.q) > 0:
            top = self.mq.q[0]
            while len(self.mq.q) > 0 and top.rel_time == self.clock:
                top = self.mq.pop()
                if top.nbr_node:
                    self.unlock_nbrs(top)
                else:
                    self.unlock(top)
                if len(self.mq.q) > 0:
                    top = self.mq.q[0]

    def update_time(self):
        self.clock = (self.clock + 1) % tmax

    def get_data(self):
        '''
        Returns a tuple of
        in_state, out_state, pf_state
        '''
        data = (self.in_state, self.out_state, self.pf_state)
        return data

    def get_action(self, train):
        '''
        Get an action from an Agent
        action : (train,in_path,out_path,pf,delay)
        '''
        data = self.get_data()
        action = self.agent.act(train, data, self.tq)
        return action

    def calculate_reward(self, sched_train):
        if sched_train[-1] == 1:
            sched_train[0].reward += -100
            return -100
        elif sched_train[3] in sched_train[0].data["pref_pf"]:
            sched_train[0].reward += 100
            return 100
        else:
            sched_train[0].reward += 50
            return 50

    def lock_incompat_paths(self, train, in_path_id, out_path_id, stop):
        in_dir = train.data["in_dir"]
        out_dir = train.data["out_dir"]
        nbrs_in = PIG.neighbors(f"{in_dir}-{in_path_id}")
        nbrs_out = PIG.neighbors(f"{out_dir}-{out_path_id}")
        for in_path in nbrs_in:
            n1 = PIG.nodes[in_path]["idx"]
            res = (n1, -1, -1, stop)
            self.lock_nbrs(res)
        for out_path in nbrs_out:
            n1 = PIG.nodes[out_path]["idx"]
            res = (-1, n1, -1, stop)
            self.lock_nbrs(res)

    def update_returns(self):
        ep_return = 0
        for i in range(len(self.schedule)-1, -1, -1):
            train_id = self.schedule[i][0].data["name"]
            in_path_id = self.schedule[i][1]
            out_path_id = self.schedule[i][2]
            net_reward = ep_return + self.schedule[i][0].reward
            self.ep_returns.append(
                (train_id, in_path_id, out_path_id, net_reward))

    def step(self):
        '''
        sched_train : (train,in_path_id,out_path_id,pf,arr_t,dep_t,is_delayed)
        '''
        if not len(self.tq.q):
            self.update_time()
            self.check_msg()
            return 0
        if not self.tq.q[0].data["arr_t"] == self.clock:
            self.update_time()
            self.check_msg()
        # Check if there is a train that needs to be scheduled at this time step
        if self.tq.q[0].data["arr_t"] == self.clock:
            train = self.tq.pop()
            action = self.get_action(train)
            if action[-1]:
                sched_train = (action[0], 0, 0, 0, 0, 0, 1)
                self.calculate_reward(sched_train)
                action[0].delay()
                self.net_delay += 1
                self.tq.push(action[0])
                print(f"Train {action[0].data['name']} is delayed by 1 min")

            else:
                sched_train = (action[0], action[1], action[2], action[3], action[0].data["arr_t"],
                               action[0].get_dep_time(), 0)
                self.schedule.append(sched_train)
                res = (sched_train[1], sched_train[2],
                       sched_train[3], sched_train[0].data["stop"])
                self.lock(res)
                self.lock_incompat_paths(
                    sched_train[0], sched_train[1], sched_train[2], sched_train[0].data["stop"])
                print(
                    f"in_path {self.in_state} out_path:{self.out_state} pf: {self.pf_state}")
                # print(len(self.mq.q))
                print(
                    f"Train {sched_train[0].data['name']} in_path: {arr_paths[sched_train[1]]},out_path: {dep_paths[sched_train[2]]},pf: {sched_train[3]} arr_t: {sched_train[4]} dep_t: {sched_train[5]}")
                # print(f"Train {sched_train[0].data['name']} in_path: {arr_paths_id[sched_train[1]]}, out_path: {dep_paths_id[sched_train[2]]},
                # pf: {sched_train[3]} arr_t: {sched_train[4]} dep_t: {sched_train[5]}")
                # print(self.schedule[-1])
                self.calculate_reward(sched_train)
                # self.agent.get_reward(reward)
                # return reward
        else:
            return 0


def run_episode(agent, env, init_conds, n_steps):
    env.reset(init_conds)
    env.init_agent(agent)
    for i in range(n_steps):
        print(f"Env Step no. {i} Clock {env.clock}")
        env.step()
    env.update_returns()
    agent.update_q_table(env.ep_returns)
    for i in range(len(env.schedule)):
        print(f"{env.schedule[i][0].data['name']} {env.schedule[i][1:]}")


if __name__ == "__main__":
    '''
    init_conds  : A dictionary with keys
                    "tlist"      : (list of trains),
                    "start_time" :(start time of clock)
                    "in_state"   : initial state of inbound paths
                    "out_state"  : initial state of outbound
                    "pf_state"   : initial state of platforms
    '''
    tlist = read_trains("train_data.txt")
    init_conds = {
        "tlist": tlist,
        "mlist": [],
        "start_time": 0,
        "in_state": [],
        "out_state": [],
        "pf_state": []
    }
    # agent = Agent_MCTS()
    env = Env(init_conds)
    agent = Agent_MCTS(tlist, env.in_state, env.out_state)
    epochs = 1
    for _ in range(epochs):
        run_episode(agent, env, init_conds, 50)
    curr_date = f"{datetime.now().strftime(time_format+'-'+date_format)}"
    save_agent(
        agent, f"MCTS-{curr_date}")
    # file = open(f"MCTS-{curr_date}.pickle", 'rb')
    agent = load_agent(f"MCTS-{curr_date}")
    # print(datetime.now())
    # print(list(PIG.nodes))
    # nx.draw_shell(PIG, with_labels=True, labels=PIG_labels)
    # plt.show()
    # print(arr_paths)
    # print(dep_paths)
    # print(arr_paths[0][-1])
    # print(PIG.nodes)
