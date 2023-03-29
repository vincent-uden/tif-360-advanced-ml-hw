import argparse
import datetime

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import font_manager
from scipy.ndimage import uniform_filter1d


def plot_reward(path: str, title: str):
    with open(path) as f:
        contents = f.readlines()
        data = []
        for row in contents:
            cells = [ float(x) for x in row.split(",") ]
            data.append(cells)

    data = np.array(data)
    episodes = np.arange(data.shape[0]) + 1

    flist = font_manager.findSystemFonts()
    names = font_manager.get_font_names()
    for n in names:
        # print(n)
        pass

    plt.plot(episodes, data[:,0])
    plt.plot(episodes, uniform_filter1d(data[:,0], size=100))
    plt.xlabel("$E$")
    plt.ylabel("$R$")
    plt.title(title)
    if title == "":
        plt.savefig(f"./cache/plots/{datetime.datetime.now().isoformat()}.pdf")
    else:
        plt.savefig(f"./cache/plots/{title}.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plot Q-Learning Data",
    )
    parser.add_argument("path")
    parser.add_argument("-m", "--mode", choices=["reward"], default="reward")
    parser.add_argument("-t", "--title", default="")

    args = parser.parse_args()

    match args.mode:
        case "reward":
            plot_reward(args.path, args.title)
