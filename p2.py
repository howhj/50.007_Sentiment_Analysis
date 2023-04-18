#Q1
        

def _construct_transition_table(training_file):
    
    transtable = {}
    hidden_states_list = []
    prev_y = "START" 
    with open(training_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                if prev_y == "STOP":
                    prev_y == "START"
                current_y = temp[1]
                if not prev_y in transtable:
                    transtable[prev_y] = {"count": 1}
                else:
                    transtable[prev_y]["count"] += 1
                if not current_y in transtable[prev_y]:
                    transtable[prev_y][current_y] = 1
                else:
                    transtable[prev_y][current_y] += 1
                if prev_y not in hidden_states_list:
                    hidden_states_list.append(prev_y)     
                prev_y = current_y              
            else:
                current_y = "STOP"
                if not prev_y in transtable:
                    transtable[prev_y] = {"count": 1}
                else:
                    transtable[prev_y]["count"] += 1
                if not current_y in transtable[prev_y]:
                    transtable[prev_y][current_y] = 1
                else:
                    transtable[prev_y][current_y] += 1
                if prev_y not in hidden_states_list:
                    hidden_states_list.append(prev_y) 
                prev_y = current_y

    return transtable, hidden_states_list    

def transition(x, y, transtable):
    return transtable[x][y] / transtable[x]["count"]
              

#Q2
import numpy as np
import p1
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
    final_path = []
    for t in range(len(obs_list)-1, 0, -1):
        best_path.append(best_parent[best_path[-1], t])
    best_path.reverse()
    for state in best_path:
        final_path.append[states_list[state]]

    return final_path

# Todo: 
# Create obs_list, states_list, trans_dict and emit_dict from previous question parts
emit_dict, obs_list = p1.construct_emission_table(k, training_file)
trans_dict, states_list = _construct_transition_table(training_file)
# Run viterbi algorithm to get all the tags
def viterbi_implement(emit_dict, trans_dict, states_list, wordlist, testing_file, output_file):
    obs_list = []
    tag = False
    append_path = []
    tagged = []
    with open(testing_file, "r") as f:
        for line in f:
            word = line.rstrip()
            if word == "":
                if tag == False:
                    obs_list.append("START")
                    tag = True
                elif tag == True:
                    index = 0
                    append_path = viterbi(obs_list, states_list, trans_dict, emit_dict)
                    for i in range(len(append_path)):
                        tagged.append(f"{obs_list[index]} {append_path[i]}\n")
                        index = index + 1
                    tagged.append("\n")
                    obs_list.clear()
                    obs_list.append("START")
                continue
            elif word in wordlist:
                obs_list.append(word)
            else:
                obs_list.append("#UNK#")

    with open(output_file, "w") as fout:
        fout.writelines(tagged)
        
# Report the precision, recall and F scores of all systems
# Fix possible numerical underflow
