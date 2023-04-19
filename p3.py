import numpy as np
import argparse
from p1 import construct_emission_table
from p2 import log_transition, log_emission, inf_sum, main

def construct_transition_table_2(training_file):
    # States are named t -> u -> v, so we want P(v | u, t)
    ttable = {}
    states = []
    u = "START"
    t = "START"

    with open(training_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                v = temp[1]
                ttable_append_2(t, u, v, ttable, states)
                t = u
                u = v

            else:
                ttable_append_2(t, u, "STOP", ttable, states)
                u = "START"
                t = "START"

    return ttable, states

def ttable_append_2(t, u, v, ttable, states):
    if (t, u) not in ttable:
        ttable[(t, u)] = {"count": 1}
    else:
        ttable[(t, u)]["count"] += 1

    if v not in ttable[(t, u)]:
        ttable[(t, u)][v] = 1
    else:
        ttable[(t, u)][v] += 1

    if u not in states and u != "START" and u != "STOP":
        states.append(u)

def viterbi_2(seq, states, ttable, etable):
    # Forward process
    # Init
    n = len(seq)
    m = len(states)

    pi = np.zeros((n+3, m+1)) # Reserve [:, m] for "START" and "STOP"
    pi[0, m] = 1 # "START"
    pi[1, m] = 1 # The cooler "START"

    # Manually do the first iteration due to the special row assignment for "START"
    for v in range(m):
        pi[2, v] = inf_sum(pi[0, m], #
                          log_transition(("START", "START"), states[v], ttable),
                          log_emission(seq[0], states[v], etable))

    # Iteration
    for j in range(1, n):
        for v in range(m):
            probs = []
            for u in range(m):
                # Pick one, both within margin of error:
                # Idea 1: best path up to t, then combine with transition from t to v
                #for t in range(m):
                #    probs.append(inf_sum(log_transition((states[t], states[u]), states[v], ttable), pi[j-1, t]))

                # Idea 2: best path up to u, then combine with best transition from t to v via u
                t_to_u_to_v = [log_transition((states[u], states[t]), states[v], ttable) for t in range(m)]
                probs.append(inf_sum(pi[j, u], np.max(t_to_u_to_v)))
            pi[j+2, v] = inf_sum(np.max(probs), log_emission(seq[j], states[v], etable))

    # End
    probs = []
    for u in range(m):
        #for t in range(m):
        #    probs.append(inf_sum(log_transition((states[t], states[u]), "STOP", ttable), pi[j-1, t]))

        t_to_u_to_v = [log_transition((states[t], states[u]), "STOP", ttable) for t in range(m)]
        probs.append(inf_sum(pi[j, u], np.max(t_to_u_to_v)))

    pi[n+2, m] = np.max(probs)


    # Backtracking
    y = [None for _ in range(n)]
    probs = []
    for u in range(m):
        probs.append(inf_sum_array([inf_sum(pi[n, u],
                                            log_transition((states[t], states[u]), "STOP", ttable))
                                   for t in range(m)]))
    y[n-1] = states[np.argmax(probs)]

    for j in range(n-2, -1, -1):
        y[j] = states[np.argmax([inf_sum(pi[j+2, u],
                                         np.max([log_transition((states[t], states[u]), y[j+1], ttable) for t in range(m)]))
                                for u in range(m)])]
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Part 3")
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
    ttable, states = construct_transition_table_2(training_file)
    main(wordlist, states, ttable, etable, testing_file, output_file, viterbi_2)
