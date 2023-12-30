# -*- coding: utf-8 -*-
import itertools

list1 = ['6_1', '6_2', '6_3']
list2 = ['13_1', '6_2', '6_3']
list3 = ['22_1', '22_2', '22_3']
list4 = ['30_1', '30_2', '30_3']
list5 = ['42_1', '42_2', '42_3']



# Create a list of all your lists
all_lists = [list1, list2, list3, list4, list5]

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
print(len(a))