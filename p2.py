#Q1

def _construct_transition_table(training_file):
    
    etable = {}
    prev_y = "START"
    with open(training_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            if len(temp) == 2:
                if prev_y == "STOP":
                    prev_y == "START"
                y = temp[1]
                if not y in etable:
                    etable[y] = {"count": 1}
                else:
                    etable[y]["count"] += 1
                if not prev_y in etable[y]:
                    etable[y][prev_y] = 1
                else:
                    etable[y][prev_y] += 1
                prev_y = temp[1]
            else:
                y = "STOP"
                if not y in etable:
                    etable[y] = {"count": 1}
                else:
                    etable[y]["count"] += 1
                if not prev_y in etable[y]:
                    etable[y][prev_y] = 1
                else:
                    etable[y][prev_y] += 1
                prev_y = "STOP"
                

def transition(x, y, etable):
    
    numerator = 0
    for k, v in etable.items():
        if k == y:
            for z,t in etable[k].items():
                if z == x:
                    numerator = t
                break
        break
    return numerator / etable[y]["count"]              



