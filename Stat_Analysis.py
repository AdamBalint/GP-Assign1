from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
# Get all the files
files = [f for f in listdir("Logs/") if isfile(join("Logs/", f))]

# set the testing p-value
p_value = 0.05

# Hold the results
results = []

# open an stats summary and a logs summary to write to
output = open("Logs/Stats/Summary.txt", 'w')
summary_logs = open("Logs/Summary.txt", 'w')

# Write the header for the stats summary
output.write("------------Normality Test------------\n")
output.write("-----File-----Normal Distribution-----\n")

# loop through all the files
for f in files:
    # if the log file is an avg log file then open it
    if ("avg" in f):
        inp = open("Logs/" + f, 'r')
        name = str(inp.name).split("/")[1].split(".")[0]

        # Store all of the results for the % classification and the type
        avg = []
        tp, tn, fp, fn = 0,0,0,0
        for line in inp:
            tmp = line.strip('\n')
            tmp2 = tmp.split("\t")
            avg.append(float(tmp2[0]))
            tp += int(tmp2[1])
            tn += int(tmp2[2])
            fp += int(tmp2[3])
            fn += int(tmp2[4])

        # calculate the total for each type of classification
        tot = tp+tn+fp+fn

        print("tp: " + str(tp/tot) + "\ttn: " + str(tn/tot))
        print("fp: " + str(fp/tot) + "\tfn: " + str(fn/tot))
        # Write the average of the classifications to the summary file and
        # write the % of each type of classification
        summary_logs.write(name + "\t" + str(np.mean(avg)) + "\t")
        summary_logs.write(str(tp/tot) + "\t" + str(tn/tot)+"\t")
        summary_logs.write(str(fp/tot) + "\t" + str(fn/tot)+"\n")

        # H0: data normally distributed
        # H1: data not normally distributed
        results.append(avg)
        # Check if data is normally distributed or not
        res = stats.mstats.normaltest(avg)
        print (f + " Results: \nPassed: " + ("Yes\n" if (res[1] > p_value) else "No\n") + str(res) + "\n")
        # if the result value is > than the pvalue we accept the null hypothesis
        # write result to the file
        output.write(name + "," + ("Yes\n" if (res[1] > p_value) else "No\n"))
        # Plot and save boxplots for the test
        plt.figure()
        plot1 = plt.boxplot(avg,
                    vert=False,         # creates horizontal box
                    widths = 0.2,       # width of the box
                    patch_artist=False)  # enable further customizatons
        plt.savefig("Graphs/Stats/bp" + name + ".png")
        plt.clf()
        # Generate and save a p-p plot for the experiment
        res_plot = stats.probplot(avg, plot=plt)
        plt.savefig("Graphs/Stats/pp" + name + ".png")

        # Generate and save a histogram for the experiment
        plt.clf()
        hist = np.histogram(avg, bins='fd')
        plt.hist(hist, bins='auto')
        plt.savefig("Graphs/Stats/hist-" + name + ".png")
        print("Res1: " + str(res[1]))

        # If the result is less than the p-value do a box-cox transformation to
        # try to correct for normality
        if (res[1] < p_value):
            bc, _ = stats.boxcox(avg)
            res2 = stats.mstats.normaltest(bc)
            print (f + " Results2: \nPassed: " + ("Yes\n" if (res2[1] <= p_value) else "No\n") + str(res2) + "\n")
            plt.clf()
            res = stats.probplot(bc, plot=plt)
            plt.savefig("Graphs/Stats/bc-pp-" + name + ".png")
        inp.close()

# Write the 2nd set of headers for the stat summary
output.write("--------------------------------------\n")
output.write("--------Statistical Difference--------\n")
output.write("--------------------------------------\n")

output.write(","+(",".join(["Test "+str(x) for x in range(len(results))])))
output.write("\n")

print("Results: " + str(len(results)))

# Loop through and check each pair of results to see if they are different from
# each other using a Mann-Whitney U test, and save the result to the file
for a in range(len(results)):
    output.write("Test" + str(a) + ",")
    output.write(","*(a+1))
    for b in range ((a+1),len(results)):

        test_res = stats.mannwhitneyu(results[a], results[b], alternative='two-sided')
        output.write(str(test_res[1]))
        if (b < (len(results)-1)):
            output.write(",")
    output.write("\n")


output.close()
