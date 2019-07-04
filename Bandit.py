import matplotlib.pyplot as plt
from matplotlib import backend_bases
from matplotlib.backends.backend_pgf import FigureCanvasPgf

import numpy as np
from tqdm import tqdm

Q_STAR = np.random.normal(size=10)
LEVER_NUM = 10
EPS = [.1, .01, .009]


def main():
    opt = np.argmax(Q_STAR)
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    # https://matplotlib.org/users/pgf.html
    # backend_bases.register_backend('pdf', FigureCanvasPgf)
    ax1.set_ylabel('Average reward')
    ax1.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal action')
    ax2.set_xlabel('Steps')

    for eps in EPS:
        result = np.zeros(1000)
        bst_action = np.zeros(1000)
        for it_outer in tqdm(range(2000)):
            Q = np.zeros(LEVER_NUM)
            N = np.zeros(LEVER_NUM)
            for it_inner in range(1000):
                a = np.random.choice([optimize_reward(Q), explore()], p=[1 - eps, eps])
                r = pull_lever(a)
                bst_action[it_inner] += (a == opt)
                N[a] += 1
                Q[a] += (1 / N[a]) * (r - Q[a])
                result[it_inner] += r
        result /= 2000
        bst_action /= 2000
        ax1.plot(result, label='ε=' + str(eps))
        ax2.plot(bst_action, label='ε=' + str(eps))
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    plt.show()


def pull_lever(action_number):
    return np.random.normal(loc=Q_STAR[action_number])


def optimize_reward(q):
    return np.argmax(q)


def explore():
    return np.random.choice(range(LEVER_NUM))


# Main Hook
if __name__ == "__main__":
    main()
