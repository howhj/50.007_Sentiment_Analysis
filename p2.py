import numpy as np
import argparse
import p1

#Q1
def _construct_transition_table(k, training_file):
    
    transtable = {}
    hidden_states_list = ["START", "STOP"] #, "#UNK#"]
    special_states = ("START", "STOP")
    u = "START"

    with open(training_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                v = temp[1]
                if not u in transtable:
                    transtable[u] = {"count": 1} # + k, "#UNK#": 1}
                else:
                    transtable[u]["count"] += 1

                if not v in transtable[u]:
                    transtable[u][v] = 1
                else:
                    transtable[u][v] += 1

                if u not in hidden_states_list and not u in special_states:
                    hidden_states_list.append(u)
                u = v

            else:
                v = "STOP"
                if not u in transtable:
                    transtable[u] = {"count": 1} # + k, "#UNK#": 1}
                else:
                    transtable[u]["count"] += 1

                if not v in transtable[u]:
                    transtable[u][v] = 1
                else:
                    transtable[u][v] += 1

                if u not in hidden_states_list and not u in special_states:
                    hidden_states_list.append(u)
                u = "START"

    return transtable, hidden_states_list    

def log_transition(u, v, transtable):
    if v == "START" or u == "STOP":
        return np.NINF
    try:
        prob = transtable[u][v] / transtable[u]["count"]
    except KeyError:
        return np.NINF
    return np.log(prob) if prob != 0 else np.NINF

#Q2
def viterbi(obs_list, states_list, trans_dict, emit_dict, seq):
    # Forward process
    # Init
    n = len(seq) + 1
    m = len(states_list)

    pi = np.zeros((n+2, m+1))
    pi[0, m] = 1 # "START"

    # Iteration
    for j in range(len(seq)):
        for v in range(m):
            pi[j+1, v] = np.max([inf_sum(pi[j, u]
                                         + log_transition(states_list[u], states_list[v], trans_dict)
                                         + log_emission(seq[j], states_list[v], emit_dict))
                                for u in range(m)])

    # End
    pi[n+1, m] = np.max([inf_sum(pi[n, u]
                                 + log_transition(states_list[u], "STOP", trans_dict))
                        for u in range(m)]) # "STOP"


    # Backtracking
    y = [None for _ in range(n+1)]
    y[n] = states_list[np.argmax([inf_sum(pi[n, u]
                                          + log_transition(states_list[u], "STOP", trans_dict))
                                 for u in range(m)])]

    for j in range(n-1, -1, -1):
        y[j] = states_list[np.argmax([inf_sum(pi[j, u]
                                              + log_transition(states_list[u], y[j+1], trans_dict))
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
def viterbi_implement(emit_dict, trans_dict, states_list, wordlist, testing_file, output_file):
    seq = []
    true_seq = []
    tagged = []
    with open(testing_file, "r") as f:
        for line in f:
            word = line.rstrip()

            if word == "":
                if seq != []:
                    tags = viterbi(wordlist, states_list, trans_dict, emit_dict, seq)
                    for w, t in zip(true_seq, tags):
                        tagged.append(f"{w} {t}\n")
                    seq = []
                    true_seq = []
                tagged.append("\n")
                continue

            elif word in wordlist:
                x = word
            else:
                x = "#UNK#"
            seq.append(x)
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

    etable, wordlist = p1.construct_emission_table(k, training_file)
    trans_dict, states_list = _construct_transition_table(k, training_file)
    viterbi_implement(etable, trans_dict, states_list, wordlist, testing_file, output_file)

# Todo:

# Report the precision, recall and F scores of all systems -> evalScript
# Fix possible numerical underflow -> log likelihood
