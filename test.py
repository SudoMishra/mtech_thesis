from datetime import datetime
from datetime import timedelta

fmt = '%H:%M:%S'
t1 = datetime.strptime('09:05:00', fmt)
t2 = datetime.strptime('09:10:00', fmt)

td = t1-t2
# print(td.total_seconds()/60)
dt = timedelta(minutes=10)
t1 = t1+dt
print(type(t2))
print(type(td))


def create_dep_paths(fname):
    file = open('arr_paths/CNB_arr_'+fname+'.txt', 'r')
    out = open('dep_paths/CNB_dep_'+fname+'.txt', 'w')
    for line in file.readlines():
        line = line.rstrip().split()
        line = line[::-1]
        out.writelines(' '.join(line)+'\n')
    out.close()
    file.close()


# if __name__ == "__main__":
#     # create_dep_paths('D4')
