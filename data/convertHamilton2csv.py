import csv





tree = open('../data/tree2_hamilton', 'r')
final = []
while True:
    branch = tree.readline()
    if branch:
        branch = branch.strip()
        line_split = branch.split()

        if len(line_split) == 2:

            childindex = line_split[1]
            parentindex = line_split[0]

            if parentindex == '2619':
                continue
            final.append([childindex, parentindex, 1])

    else:
        break



print(final)
with open('../data/hamilton_closure.csv', 'w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(['id1','id2','weight'])
    f_csv.writerows(final)
