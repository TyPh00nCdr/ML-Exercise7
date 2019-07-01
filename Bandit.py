import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

Q_STAR = np.random.normal(size=10)
LEVER_NUM = 10
EPS = [.1, .01, .009]


def main():
    eps = EPS[0]
    result = np.zeros(1000)

    for it_outer in tqdm(range(2000)):
        Q = np.zeros(LEVER_NUM)
        N = np.zeros(LEVER_NUM)
        for it_inner in range(1000):
            a = np.random.choice([optimize_reward(Q), explore()], p=[1 - eps, eps])
            r = pull_lever(a)
            N[a] += 1
            Q[a] += (1 / N[a]) * (r - Q[a])
            result[it_inner] += r

    result /= 2000
    result = np.cumsum(result) / np.arange(start=1, stop=1001)
    plt.plot(result)
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
