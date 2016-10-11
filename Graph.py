import matplotlib.pyplot as plt


def plotGraph(name, gen, y1, y2):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gen, y1)
    ax.plot(gen, y2)


    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(name)
    plt.grid(True)

    fig.savefig("Graphs/"+name+".png")
