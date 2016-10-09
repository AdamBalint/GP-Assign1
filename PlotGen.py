import matplotlib.pyplot as plt
import numpy as np


f = open("Logs/test-Logs-2.txt")

lines = f.readlines()

logs = []

gen = []
y1 = []
y2 = []


counter = 0
for line in lines:
    tmp = (line.split("\t"))
    print(tmp)
    if (not(counter < 3)):
        gen.append(int(tmp[0]))
        y1.append(float(tmp[3]))
        y2.append(float(tmp[7]))

        print(tmp)
        logs.append(tmp)
    counter += 1



plt.plot(gen, y1)
plt.plot(gen, y2)



plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('GP Run')
plt.grid(True)
plt.savefig("testGP.png")
plt.show()
