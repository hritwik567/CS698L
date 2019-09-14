import matplotlib.pyplot as plt
from subprocess import Popen, PIPE

p1 = [(4096, 2), (4096, 4), (4096, 8), (4096, 16)]
p2 = [(256, 1), (512, 2), (512, 4), (1024, 8), (1024, 16), (2048, 16), (4096, 16)]
s1 = []
s2 = []
out = dict()

f = open("outputs", "r+")
data = f.read()[:-1].split("\n\n")
f.close()

for i in data:
  t = i.split('\n')
  out[(int(t[1].split('=')[-1]), int(t[2].split('=')[-1]))] = (float(t[-3].split('=')[-1][:-3]), float(t[-2].split('=')[-1]))

s1 = [out[i][1] for i in p1]
s2 = [out[i][0] for i in p2]

print(s1)
print(s2)

s2 = list(map(lambda x: s2[0]/x, s2))

p1 = list(map(lambda x: "N:" + str(x[0]) + ", Th:" + str(x[1]), p1))
p2 = list(map(lambda x: "N:" + str(x[0]) + ", Th:" + str(x[1]), p2))

plt.xticks(range(1, len(s1) + 1), p1)
plt.ylabel("SpeedUp w.r.t sequential version")
[plt.annotate("%0.4f" % i[1], i) for i in zip(range(1, len(s1) + 1), s1)]
plt.plot(range(1, len(s1) + 1), s1, "go-")
plt.show()
plt.clf()

plt.xticks(range(1, len(s2) + 1), p2)
plt.ylabel("SpeedUp w.r.t N:256, Th:1")
[plt.annotate("%0.4f" % i[1], i) for i in zip(range(1, len(s2) + 1), s2)]
plt.plot(range(1, len(s2) + 1), s2, "go-")
plt.show()
plt.clf()
