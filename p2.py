import numpy as np
import argparse
from p1 import construct_emission_table

#Q1
def construct_transition_table(training_file):
    ttable = {}
    states = []
    u = "START"

    with open(training_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                v = temp[1]
                ttable_append(u, v, ttable, states)
                u = v

            else:
                ttable_append(u, "STOP", ttable, states)
                u = "START"

    return ttable, states

def ttable_append(u, v, ttable, states):
    if u not in ttable:
        ttable[u] = {"count": 1}
    else:
        ttable[u]["count"] += 1

    if v not in ttable[u]:
        ttable[u][v] = 1
    else:
        ttable[u][v] += 1

    if u not in states and u != "START" and u != "STOP":
        states.append(u)

def log_transition(u, v, ttable):
    if v == "START" or u == "STOP":
        return np.NINF
    try:
        prob = ttable[u][v] / ttable[u]["count"]
    except KeyError:
        return np.NINF
    return np.log(prob) if prob != 0 else np.NINF

#Q2
def viterbi(seq, states, ttable, etable):
    # Forward process
    # Init
    n = len(seq)
    m = len(states)

    pi = np.zeros((n+2, m+1)) # Reserve [:, m] for "START" and "STOP"
    pi[0, m] = 1 # "START"

    # Manually do the first iteration due to the special row assignment for "START"
    for v in range(m):
        pi[1, v] = inf_sum(pi[0, m],
                          log_transition("START", states[v], ttable),
                          log_emission(seq[0], states[v], etable))

    # Iteration
    for j in range(1, n):
        for v in range(m):
            pi[j+1, v] = np.max([inf_sum(pi[j, u],
                                         log_transition(states[u], states[v], ttable),
                                         log_emission(seq[j], states[v], etable))
                                for u in range(m)])

    # End
    pi[n+1, m] = np.max([inf_sum(pi[n, u],
                                 log_transition(states[u], "STOP", ttable))
                        for u in range(m)]) # "STOP"


    # Backtracking
    y = [None for _ in range(n)]
    y[n-1] = states[np.argmax([inf_sum(pi[n, u],
                                       log_transition(states[u], "STOP", ttable))
                              for u in range(m)])]

    for j in range(n-2, -1, -1):
        y[j] = states[np.argmax([inf_sum(pi[j+1, u],
                                         log_transition(states[u], y[j+1], ttable))
                                for u in range(m)])]
    return y

# Helper functions
def log_emission(x, y, etable):
    if y == "START" or y == "STOP":
        return np.NINF
    try:
        prob = etable[y][x] / etable[y]["count"]
    except KeyError:
        return np.NINF
    return np.log(prob) if prob != 0 else np.NINF

def inf_sum(*args):
    total = 0
    for arg in args:
        if np.isinf(arg):
            return np.NINF
        total += arg
    return total

# Run viterbi algorithm to get all the tags
def main(wordlist, states, ttable, etable, testing_file, output_file, viterbi_fn):
    seq = []
    true_seq = []
    tagged = []

    with open(testing_file, "r") as f:
        for line in f:
            word = line.rstrip()

            # Sequence ended, run Viterbi on the sequence we have
            if word == "":
                if seq != []:
                    tags = viterbi_fn(seq, states, ttable, etable)
                    for w, t in zip(true_seq, tags):
                        tagged.append(f"{w} {t}\n")
                    seq = []
                    true_seq = []
                tagged.append("\n")
                continue

            # Otherwise, keep adding the token to the sequence
            elif word in wordlist:
                seq.append(word)
            else:
                seq.append("#UNK#")
            true_seq.append(word)

    with open(output_file, "w") as fout:
        fout.writelines(tagged)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Part 2")
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
    ttable, states = construct_transition_table(training_file)
    main(wordlist, states, ttable, etable, testing_file, output_file, viterbi)
