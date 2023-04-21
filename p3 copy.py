import numpy as np
import argparse
from p1 import construct_emission_table
from p2 import log_transition, log_emission, inf_sum, construct_transition_table

def construct_transition_table_2(training_file):
    # States are named t -> u -> v, so we want P(v | u, t)
    ttable = {}
    u = "START"
    t = "START"

    with open(training_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                v = temp[1]
                ttable_append_2(t, u, v, ttable)
                t = u
                u = v

            else:
                ttable_append_2(t, u, "STOP", ttable)
                u = "START"
                t = "START"

    return ttable

def ttable_append_2(t, u, v, ttable):
    if (t, u) not in ttable:
        ttable[(t, u)] = {"count": 1}
    else:
        ttable[(t, u)]["count"] += 1

    if v not in ttable[(t, u)]:
        ttable[(t, u)][v] = 1
    else:
        ttable[(t, u)][v] += 1

def viterbi_2(seq, states, ttable, ttable2, etable):
    # Forward process
    # Init
    n = len(seq)
    m = len(states)

    pi = np.zeros((n+3, m+1)) # Reserve [:, m] for "START" and "STOP"
    pi[0, m] = 1 # "START"
    pi[1, m] = 1 # The cooler "START"

    # Manually do the first iteration due to the special row assignment for "START"
    for v in range(m):
        order1 = inf_sum(pi[1, m], log_transition("START", states[v], ttable))
        order2 = inf_sum(pi[0, m], log_transition(("START", "START"), states[v], ttable2))
        pi[2, v] = inf_sum(order1, order2, log_emission(seq[0], states[v], etable))

    # Iteration
    for j in range(1, n):
        for v in range(m):
            o1 = []
            for u in range(m):
                order1 = inf_sum(pi[j+1, u], log_transition(states[u], states[v], ttable))
                o2 = []
                for t in range(m):
                    order2 = inf_sum(pi[j, t], log_transition((states[t], states[u]), states[v], ttable2))
                    o2.append(order2)
                o1.append(inf_sum(np.max(o2), order1))
            pi[j+2, v] = inf_sum(np.max(o1), log_emission(seq[j], states[v], etable))

    # End
    o1 = []
    for u in range(m):
        order1 = inf_sum(pi[n+1, u], log_transition(states[u], "STOP", ttable))
        o2 = []
        for t in range(m):
            order2 = inf_sum(pi[n, t], log_transition((states[t], states[u]), "STOP", ttable2))
            o2.append(order2)
        o1.append(inf_sum(np.max(o2), order1))
    pi[n+2, m] = np.max(o1)


    # Backtracking
    y = [None for _ in range(n)]
    #probs = []
    #for u in range(m):
    #    t_to_v = [log_transition((states[t], states[u]), "STOP", ttable) for t in range(m)]
    #    probs.append(inf_sum(pi[n+1, u], np.max(t_to_v)))
    y[n-1] = states[np.argmax(o1)]

    for j in range(n-2, -1, -1):
        o1 = []
        for u in range(m):
            order1 = inf_sum(pi[j+2, u], log_transition(states[u], y[j+1], ttable))
            o2 = []
            for t in range(m):
                order2 = inf_sum(pi[j+1, t], log_transition((states[t], states[u]), y[j+1], ttable2))
                o2.append(order2)
            o1.append(inf_sum(np.max(o2), order1))
        y[j] = states[np.argmax(o1)]

    return y

def main(wordlist, states, ttable, ttable2, etable, testing_file, output_file, viterbi_fn):
    seq = []
    true_seq = []
    tagged = []

    with open(testing_file, "r") as f:
        for line in f:
            word = line.rstrip()

            # Sequence ended, run Viterbi on the sequence we have
            if word == "":
                if seq != []:
                    tags = viterbi_fn(seq, states, ttable, ttable2, etable)
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
    ttable, states = construct_transition_table(training_file)
    ttable2 = construct_transition_table_2(training_file)
    main(wordlist, states, ttable, ttable2, etable, testing_file, output_file, viterbi_2)
