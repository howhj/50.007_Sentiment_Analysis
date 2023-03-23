import argparse

# Q1, no #UNK#
# Parse file and construct emission table first
# Then look up values from it to calculate probability
def _construct_emission_table(training_file):
    etable = {}
    wordlist = []
    with open(training_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                x, y = temp[0], temp[1]
                if not y in etable:
                    etable[y] = {"count": 1}
                else:
                    etable[y]["count"] += 1

                if not x in etable[y]:
                    etable[y][x] = 1
                else:
                    etable[y][x] += 1

                if x not in wordlist:
                    wordlist.append(x)
    return etable, wordlist

def emission(x, y, etable):
    return etable[y][x] / etable[y]["count"]

# Q2, with #UNK#
# Reuses the emission() function from Q1
def construct_emission_table(k, training_file):
    etable = {}
    wordlist = []
    with open(training_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                x, y = temp[0], temp[1]
                if not y in etable:
                    etable[y] = {"count": 1 + k, "#UNK#": k}
                else:
                    etable[y]["count"] += 1

                if not x in etable[y]:
                    etable[y][x] = 1
                else:
                    etable[y][x] += 1

                if x not in wordlist:
                    wordlist.append(x)
    return etable, wordlist

# Q3
def sentiment_analysis(etable, wordlist, testing_file, output_file):
    tagged = []
    probs = {}
    for k in etable.keys():
        probs[k] = 0

    with open(testing_file, "r") as f:
        for line in f:
            # Determine token
            word = line.rstrip()
            if word == "":
                tagged.append("\n")
                continue
            elif word in wordlist:
                x = word
            else:
                x = "#UNK#"

            # Calculate emission probability for each tag
            for y in probs.keys():
                if x in etable[y]:
                    probs[y] = emission(x, y, etable)
                else:
                    probs[y] = 0

            # Determine best matching tag
            max_prob = 0
            best_tag = ""
            for k, v in probs.items():
                if v > max_prob:
                    max_prob = v
                    best_tag = k

            tagged.append(f"{word} {best_tag}\n")

    with open(output_file, "w") as fout:
        fout.writelines(tagged)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Part 1")
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
    sentiment_analysis(etable, wordlist, testing_file, output_file)
