# -*- coding: utf-8 -*-
import itertools

list1 = ['A1', 'A2']
list2 = ['B1', 'B2']
list3 = ['C1', 'C2']
list4 = ['D1', 'D2']
list5 = ['E1', 'E2']
list6 = ['F1', 'F2']
list7 = ['G1', 'G2']


# Create a list of all your lists
all_lists = [list1, list2, list3, list4]

# Generate all possible combinations of three values with one value from each class
combinations_3 = list(itertools.combinations(all_lists, 3))
print(len(combinations_3))
a=[]
# Iterate over combinations and print
for combo in combinations_3:
    print(combo)
    for values in itertools.product(*combo):
        a.append(values)
        print(values)
len(a)