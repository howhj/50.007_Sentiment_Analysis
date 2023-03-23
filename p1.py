import argparse

# Q1, no #UNK#
def _emission(x, y, training_file):
    ctr = 0
    total = 0
    with open(training_file, "r") as f:
        for line in f.readlines():
            temp = line.split()
            if len(temp) == 2 and temp[1] == y:
                total += 1
                if temp[0] == x:
                    ctr += 1
    return ctr / total

# Q2, with #UNK#
def emission(x, y, k, training_file):
    ctr = k if x == "#UNK#" else 0
    total = 0
    with open(training_file, "r") as f:
        for line in f.readlines():
            temp = line.split()
            if len(temp) == 2 and temp[1] == y:
                total += 1
                if x != "#UNK#" and temp[0] == x:
                    ctr += 1
    total += k
    return ctr / total

# Q3
def parse_train(k, training_file):
    dct = {}
    lst = []
    with open(training_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                x, y = temp[0], temp[1]
                if not y in dct:
                    dct[y] = {"count": 1 + k, "#UNK#": k}
                else:
                    dct[y]["count"] += 1

                if not x in dct[y]:
                    dct[y][x] = 1
                else:
                    dct[y][x] += 1

                if x not in lst:
                    lst.append(x)
    return dct, lst

# Uses data stored in memory instead
def emission_dct(x, y, dct):
    return dct[y][x] / dct[y]["count"]

def sentiment_analysis(dct, wordlist, testing_file, output_file):
    lst = []
    probs = {}
    for k in dct.keys():
        probs[k] = 0

    with open(testing_file, "r") as f:
        for line in f:
            line = line.rstrip()
            # Determine token
            if line == "":
                lst.append("\n")
                continue
            elif line in wordlist:
                x = line
            else:
                x = "#UNK#"

            # Calculate emission probability for each tag
            for y in probs.keys():
                probs[y] = emission_dct(x, y, dct) if x in dct[y] else 0

            # Determine best matching tag
            max_prob = 0
            best_tag = "#UNK"
            for k, v in probs.items():
                if v > max_prob:
                    max_prob = v
                    best_tag = k

            lst.append(f"{line} {best_tag}\n")

    with open(output_file, "w") as fout:
        fout.writelines(lst)


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

    dct, lst = parse_train(k, training_file)
    sentiment_analysis(dct, lst, testing_file, output_file)
