import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join
# Gets all the files
files = [f for f in listdir("Logs/") if isfile(join("Logs/", f))]



# Graphs given data
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




# Loops through all the files
for f in files:
    fig_count = 0
    # If it is a logs file, then
    if ("logs" in f):
        print("Opened File: " + str(f) +"\n")
        f = open("Logs/"+f)
        # Reads the entire fie
        lines = f.readlines()

        logs = []

        gen = []
        y1 = []
        y2 = []

        fig_count += 1
        counter = 0
        # splits each line by tabs
        for line in lines:
            tmp = (line.split("\t"))
            # If there are >= 3 elements in the array
            if (not(counter < 3)):
                # append the values to lists
                gen.append(int(tmp[0]))
                y1.append(float(tmp[3]))
                y2.append(float(tmp[7]))
                logs.append(tmp)
            counter += 1
        # get the name of the file
        name = str(f.name).split("/")[1].split(".")[0]
        # plot the graph
        plotGraph(name, gen, y1, y2)
    # if too many figures are being generated in one loop then break
    if (fig_count == 10):
        break
