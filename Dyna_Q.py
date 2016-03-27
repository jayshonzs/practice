__author__ = 'xiajie'

import numpy as np
import random
import matplotlib.pyplot as plt

state_num = 54
action_num = 4
Q = np.zeros((state_num, action_num))
Model = np.zeros((state_num, action_num))
models = []
actions = [-1, 9, 1, -9]
blocks1 = set(range(28, 36))
blocks2 = set(range(28, 35))
alpha = 0.1
gamma = 0.9
epsilon = 0.1
start = 48
accumulative_rewards = []
N = 50
history_states = []
history_actions = dict()
kai = 0.007

def max_Q(s, cur_step):
    max_q = -1
    best_i = 0
    state = history_actions.get(s, {})
    for i in range(action_num):
        last_step = state.get(i, -1)
        bonus_q = Q[s, i] + kai*np.sqrt(cur_step - last_step)
        if bonus_q > max_q + 1e-8:
            max_q = bonus_q
            best_i = i
    return Q[s, best_i], best_i

def setup():
    m = 0
    for s in range(state_num):
        for i in range(action_num):
            Model[s, i] = m
            models.append([])
            m += 1

def e_greedy(s, step):
    a = max_Q(s, step)[1]
    e = random.uniform(0, 1)
    if e < 1. - epsilon:
        return a
    else:
        return random.choice(range(action_num))

def iteration(s, blocks, step):
    if s == 8:
        s = start
    if s not in history_states:
        history_states.append(s)
    history_actions.setdefault(s, {})
    a = e_greedy(s, step)
    print s, a
    history_actions[s][a] = step
    action = actions[a]
    ns = s + action
    if (s % 9 == 0 and action == -1) or (s >= 45 and action == 9) or ((s + 1) % 9 == 0 and action == 1) or (s < 9 and action == -9):
        ns = s
    if ns in blocks:
        ns = s
    r = 0.
    if ns == 8:
        r = 1.
        print ns
    Q[s, a] += alpha*(r + gamma*max_Q(ns, step)[0] - Q[s, a])
    m = int(Model[s, a])
    models[m] = (ns, r)
    if len(accumulative_rewards) == 0:
        accumulative_rewards.append(r)
    else:
        accumulative_rewards.append(accumulative_rewards[-1]+r)
    ret = ns
    for n in range(N):
        s = random.choice(history_states)
        if s in blocks:
            n -= 1
            continue
        a = random.choice(history_actions[s].keys())
        m = int(Model[s, a])
        ns, r = models[m]
        Q[s, a] += alpha*(r + gamma*max_Q(ns, step)[0] - Q[s, a])
    return ret

def plot_acc_rewards():
    print Q
    X = np.array(range(6000))
    Y = np.array(accumulative_rewards)

    plt.plot(X, Y)
    plt.xlabel('steps')
    plt.ylabel('rewards')

    plt.show()

def main():
    global accumulative_rewards
    setup()
    for loop in range(1):
        print loop
        s = start
        for i in range(3000):
            s = iteration(s, blocks1, i)
        for i in range(3000, 6000):
            s = iteration(s, blocks2, i)
        if loop == 0:
            A = np.array(accumulative_rewards)
        else:
            A += np.array(accumulative_rewards)
        accumulative_rewards = []
    accumulative_rewards = A
    plot_acc_rewards()

if __name__ == '__main__':
    main()
