import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join
files = [f for f in listdir("Logs/") if isfile(join("Logs/", f))]

#print(str(files))



def plotGraph(name, gen, y1, y2):
    print ("Creating figure: " + name)


    plt.plot(gen, y1, label="Fitness")
    plt.plot(gen, y2, label="Graph Size")


    plt.xlabel('Generation')
    plt.title(name)
    plt.grid(True)
    leg = plt.legend(loc=4)
    leg.get_frame().set_alpha(0.4)
    plt.savefig(str("Graphs/"+name+".png"))
    plt.clf()





for f in files:
    fig_count = 0
    if ("logs" in f):
        print("Opened File: " + str(f) +"\n")
        f = open("Logs/"+f)

        lines = f.readlines()

        logs = []

        gen = []
        y1 = []
        y2 = []

        fig_count += 1
        counter = 0
        for line in lines:
            tmp = (line.split("\t"))
            #print(tmp)
            if (not(counter < 3)):
                gen.append(int(tmp[0]))
                y1.append(float(tmp[3]))
                y2.append(float(tmp[7]))

            #    print(tmp)
                logs.append(tmp)
            counter += 1

        name = str(f.name).split("/")[1].split(".")[0]

        plotGraph(name, gen, y1, y2)
    if (fig_count == 10):
        break
