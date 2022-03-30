from collections import deque
import heapq
from random import randint
import gym
from gym import spaces

n_dirs = 2
npfs = 3
npriority = 4


def one_hot(data, n):
    val = [1 if i in data else 0 for i in range(n)]
    return val


# test = [1, 2, 5]
# print(f"input : {test} output : {one_hot(test,7)}")
class Agent:
    def __init__(self, in_paths, out_paths, npfs):
        self.in_act = [0 for i in range(len(in_paths))]
        self.out_act = [0 for i in range(len(out_paths))]
        # self.pf_act = [0 for i in range(len(npfs))]

    def act(self, state):
        pass


class stationEnv:
    '''
    name: Name of the Station
    npfs        : No of Platforms
    directs     : No. of directions
    in_paths    : No. of inbound paths to station
    out_paths   : No. of outbound paths to station
    init_conds  : A dictionary with keys
                    "tlist"      : (list of trains),
                    "start_time" :(start time of clock)
                    "in_state"   : initial state of inbound paths
                    "out_state"  : initial state of outbound
                    "pf_state"   : initial state of platforms  
    '''

    def __init__(self, name, npfs, directs, in_paths, out_paths, init_conds):
        self.name = name
        self.npfs = npfs
        self.directs = directs
        self.in_paths = in_paths
        self.out_paths = out_paths
        self.in_state = [0 for i in range(len(self.in_paths))]
        self.out_state = [0 for i in range(len(self.out_paths))]
        self.pf_state = [0 for i in range(npfs)]
        self.q = PriorityQueue()
        self.clock = init_conds["start_time"]
        self.init_states(init_conds)
        self.init_pq(init_conds["tlist"])

    def init_pq(self, tlist):
        for t in tlist:
            self.q.push(t)

    def init_states(self, init_conds):
        self.in_state = init_conds["in_state"]
        self.out_state = init_conds["out_state"]
        self.pf_state = init_conds["pf_state"]

    def reset(self, init_conds):
        self.init_states(init_conds)
        self.init_pq(init_conds["tlist"])
        # self.in_state = [0 for i in range(len(self.in_paths))]
        # self.out_state = [0 for i in range(len(self.out_paths))]
        # self.pf_state = [0 for i in range(self.npfs)]

        # self.clock = 0

    # def step(self, action):
    #     if action[0] == 0:
    #         self.clock = self.clock + 1
    #         self.update_q(action[1])
    #     pass

    # def update_q(train):
    #     pass


class Train:
    def __init__(self, data):
        '''
        **kwargs should have
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
        self.in_dir = one_hot(self.data["in_dir"],  n_dirs)
        self.out_dir = one_hot(self.data["out_dir"], n_dirs)
        self.pfs = one_hot(self.data["pref_pf"], npfs)
        self.prty = one_hot([self.data["prty"]], npriority)

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

    def delay(self):
        self.arr_t = (self.arr_t + 1) % 1440


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

    def pop(self, data):
        heapq.heappop(self.q)


# class Env():
#     def __init__(self):
#         self.q = PriorityQueue()


if __name__ == "__main__":
    tlist = PriorityQueue()
    for i in range(1, 4):
        data = {
            "name": f"T - {i}",
            "in_dir": [1],
            "out_dir": [1],
            "arr_t": randint(1, 7),
            "stop": 5,
            "pref_pf": [1],
            "prty": randint(1, 4)
        }
        tlist.push(Train(data))
    tlist.hsort()
    for t in tlist:
        print(t)
    # for t in tlist:
    #     print(t)
