import numpy as np
import argparse
from p1 import construct_emission_table
from p2 import log_transition, log_emission, inf_sum, main

def construct_transition_table_3(training_file):
    # States are named s -> t -> u -> v, so we want P(v | s, t, u)
    ttable = {}
    states = []
    u = "START"
    t = "START"
    s = "START"

    with open(training_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                v = temp[1]
                ttable_append_3(s, t, u, v, ttable, states)
                s, t, u = t, u, v

            else:
                ttable_append_3(s, t, u, "STOP", ttable, states)
                u = "START"
                t = "START"
                s = "START"

    return ttable, states

def ttable_append_3(s, t, u, v, ttable, states):
    if (s, t, u) not in ttable:
        ttable[(s, t, u)] = {"count": 1}
    else:
        ttable[(s, t, u)]["count"] += 1

    if v not in ttable[(s, t, u)]:
        ttable[(s, t, u)][v] = 1
    else:
        ttable[(s, t, u)][v] += 1

    if u not in states and u != "START" and u != "STOP":
        states.append(u)

def viterbi_3(seq, states, ttable, etable):
    # Forward process
    # Init
    n = len(seq)
    m = len(states)

    pi = np.zeros((n+4, m+1)) # Reserve [:, m] for "START" and "STOP"
    pi[0, m] = 1 # "START"
    pi[1, m] = 1 # The cooler "START"
    pi[2, m] = 1 # The COOLEST "START"

    # Manually do the first iteration due to the special row assignment for "START"
    for v in range(m):
        pi[3, v] = inf_sum(pi[2, m],
                          log_transition(("START", "START", "START"), states[v], ttable),
                          log_emission(seq[0], states[v], etable))

    # Iteration
    for j in range(1, n):
        for v in range(m):
            probs = []
            for u in range(m):
                s_to_v = []
                for t in range(m):
                    for s in range(m):
                        s_to_v.append(log_transition((states[s], states[t], states[u]), states[v], ttable))
                probs.append(inf_sum(pi[j+2, u], np.max(s_to_v)))
            pi[j+3, v] = inf_sum(np.max(probs), log_emission(seq[j], states[v], etable))

    # End
    probs = []
    for u in range(m):
        s_to_v = []
        for t in range(m):
            for s in range(m):
                s_to_v.append(log_transition((states[s], states[t], states[u]), "STOP", ttable))
        probs.append(inf_sum(pi[n+2, u], np.max(s_to_v)))
    pi[n+3, m] = np.max(probs)


    # Backtracking
    y = [None for _ in range(n)]
    y[n-1] = states[np.argmax(probs)]

    for j in range(n-2, -1, -1):
        probs = []
        for u in range(m):
            s_to_v = []
            for t in range(m):
                for s in range(m):
                    s_to_v.append(log_transition((states[s], states[t], states[u]), y[j+1], ttable))
            probs.append(inf_sum(pi[j+3, u], np.max(s_to_v)))
        y[j] = states[np.argmax(probs)]

    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "HMM???")
    parser.add_argument("k", type = int, help = "Smoothing factor.")
    parser.add_argument("training_file", metavar = "train", type = str, help = "Path to the file with training data.")
    parser.add_argument("testing_file", metavar = "test", type = str, help = "Path to the file with testing data.")
    parser.add_argument("output_file", metavar = "out", type = str, help = "Path to the file for storing predicted results.")

    args = parser.parse_args()
    k = args.k
    training_file = args.training_file
    testing_file = args.testing_file
    output_file = args.output_file

    etable, wordlist = construct_emission_table(k, training_file)
    ttable, states = construct_transition_table_3(training_file)
    main(wordlist, states, ttable, etable, testing_file, output_file, viterbi_3)
