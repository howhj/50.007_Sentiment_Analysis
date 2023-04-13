#Q1

def _construct_transition_table(training_file):
    
    transtable = {}
    prev_y = "START"
    with open(training_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                if prev_y == "STOP":
                    prev_y == "START"
                y = temp[1]
                if not y in transtable:
                    transtable[y] = {"count": 1}
                else:
                    transtable[y]["count"] += 1
                if not prev_y in transtable[y]:
                    transtable[y][prev_y] = 1
                else:
                    transtable[y][prev_y] += 1
                prev_y = temp[1]
            else:
                y = "STOP"
                if not y in transtable:
                    transtable[y] = {"count": 1}
                else:
                    transtable[y]["count"] += 1
                if not prev_y in transtable[y]:
                    transtable[y][prev_y] = 1
                else:
                    transtable[y][prev_y] += 1
                prev_y = "STOP"
                

def transition(x, y, transtable):
    
    numerator = 0
    for k, v in transtable.items():
        if k == y:
            for z,t in transtable[k].items(): 
                if z == x:
                    numerator = t
                break
            break
    return numerator / transtable[y]["count"]              



    
#Q2
import numpy as np
def viterbi(obs_list, states_list, trans_dict, emit_dict):
    """
    Viterbi algorithm for finding the most likely sequence of hidden states that generated a sequence of observed states

    :param obs_list: a list of observed states
    :param states_list: a list of possible hidden states
    :param trans_dict: a dict representing the transition probabilities between hidden states
    :param emit_dict: a dict representing the emission probabilities of each observed state from each hidden state
    :return: a tuple consisting of the most likely sequence of hidden states and the probability of that sequence
    """
    # Initialize the viterbi table and the best_parent (parent that gives the highest probability) table
    V = np.zeros((len(states_list), len(obs_list)))
    best_parent = np.zeros((len(states_list), len(obs_list)), dtype=int)

    # Set the initial probabilities
    # First column of viterbi table is all 0 except "START" hidden state and each hidden state best_parent is set to -1 
    # since the column is the starting state
    for i, s in enumerate(states_list):
        if (s == "START"):
            V[i, 0] = 1
        else:
            V[i,0] = 0
        best_parent[i, 0] = -1

    # Iterate through the obs_list and hidden states_list and fill up the viterbi table except the last column 
    # First column already filled above during initialising
    for t in range(1, len(obs_list) - 1):
        for j, s2 in enumerate(states_list):
            max_prob = 0
            max_index = 0
            for i, s1 in enumerate(states_list):
                prob = V[i, t-1] * trans_dict[s1][s2] * emit_dict[s2][obs_list[t]]
                if prob > max_prob:
                    max_prob = prob
                    max_index = i
            V[j, t] = max_prob
            best_parent[j, t] = max_index

    # To fill up the last column of the viterbi table, run the following code once, this is the final Step ("STOP" of HMM)
    for t in range(1):
        max_prob = 0
        max_index = 0
        for i, s1 in enumerate(states_list):
            prob = V[i, len(obs_list) - 2] * trans_dict[s1]["STOP"] 
            if prob > max_prob:
                max_prob = prob
                max_index = i
        # Last column of viterbi table is all 0 except "STOP" hidden state and each hidden state best_parent except "STOP" 
        # is set to -1 since the last column cannot take on any other hidden states except "STOP", whose best_parent is max_index
        for i, s in enumerate(states_list):
            if (s == "STOP"):
                V[i, len[obs_list]-1] = max_prob
                best_parent[len(states_list) - 1, len(obs_list) - 1] = max_index
            else:
                V[i, len[obs_list]-1] = 0
                best_parent[len(states_list) - 1, len(obs_list) - 1] = - 1
        

    # Find the final state with the highest probability (which is always "STOP" state in our case) 
    max_index = len(states_list) - 1

    # Follow the best_parent table to find the sequence of hidden states that yield the highest probability 
    best_path = [max_index]
    for t in range(len(obs_list)-1, 0, -1):
        best_path.append(best_parent[best_path[-1], t])
    best_path.reverse()

    return best_path, max_prob

# Todo: 
# Convert best_path to actual states (neutral etc) not just the index representing the state
# Create obs_list, states_list, trans_dict and emit_dict from previous question parts
# Report the precision, recall and F scores of all systems
# Fix numerical underflow