import matplotlib.pyplot as plt
import os

results_dict = {}

for root, dirs, files in os.walk("."):
    for file in files:
        if "output" in file[:len("output")]:
            file_key = file.replace("output_","")
            results_dict[file_key] = []
            with open(os.path.join(root,file)) as rf:
                for line in rf:
                    if "eval score" in line:
                        epoch, score, perfect, _ = line.strip().split(",")
                        epoch = int(epoch[epoch.index("epoch ")+len("epoch "):])
                        score = float(score[score.index("eval score: ")+len("eval score: "):])
                        perfect = float(perfect[perfect.index("perfect: ")+len("perfect: "):])
                        results_dict[file_key].append((epoch, score, perfect))


'''["dimgray","lightgray","lightsteelblue","cornflowerblue","royalblue","violet","mediumorchid","darkviolet","firebrick"]'''
colors = {
    "sad":"lightgray",
    "predtask":"lightgray",
    "pid":"lightcoral",
    # "2ffsk":"royalblue",
    "prebrief":"cornflowerblue",
    "pid_prebrief":"violet",
    # "prebrief_sqrt":"darkviolet",
}

darker_colors = {
    "lightgray":"dimgray",
    "lightcoral":"firebrick",
    "cornflowerblue":"royalblue",
    "violet":"darkviolet",
}

seen_labels = set()
for level in [5,10,15,20,21,22,23,24]:
    plt.axhline(level,linestyle="dotted",color="lightgray")
for file_key in sorted(results_dict.keys())[::-1]:
    # print(results_dict[file_key])
    x, scores, perfects = zip(*results_dict[file_key])
    color = "lightgray"
    for typekey in colors.keys():
        if typekey in file_key:
            color = colors[typekey]
            if "2ffsk" in file_key:
                color = darker_colors[color]
    # plt.xscale("symlog")
    label = "_".join(file_key.split("_")[:-1])
    plt.plot(x, scores, label=None if label in seen_labels else label, color=color)
    seen_labels.update([label])

plt.xlabel("Epochs")
plt.ylabel("Hanabi Score")
plt.legend()
plt.savefig("results.png",format='png')
plt.xlim((0,500))
plt.ylim((10,25))
plt.savefig("results_500.png",format='png')