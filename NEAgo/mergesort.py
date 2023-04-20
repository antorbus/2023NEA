def mergesort_helper(l1, l2): # a helper function that joins two sorted lists 
    if len(l1) <= len(l2): l1, l2 = l2, l1 # swaps lists so that l1 is always longer than l2
    l_merged =[]
    exp_len = len(l1) + len(l2)
    while l1: # joins both sorted lists by poping the first item from the list that has the smallest first item
        if l2:
            if l1[0] <= l2[0]: l_merged.append(l1.pop(0)) 
            else: l_merged.append(l2.pop(0))
        else:
            l_merged += l1 
            break
    l_merged += l2
    assert exp_len == len(l_merged) # checks that the length of the output is what is expected 
    return l_merged

def plain_mergesort(l): # mergesort implemented for list 
    l = [[x] for x in l] # splits list into smaller sublists
    exp_len= len(l)
    while len(l) != 1:
        for i in range(0, len(l), 2):
            sub_lists = l[i:i+2] # takes two sublists
            if len(sub_lists) ==2:
                sub_lists = mergesort_helper(*sub_lists) # sorts the two sorted sublists
                del l[i]
                l[i] = sub_lists # replaces the two sublists with a sorted one
    l = l[0]
    assert exp_len == len(l)
    return l

def mergesort(x): # wrapper for the mergesort function that allows for dictionaries to be sorted by key (polymorphism)
    try: 
        keys = plain_mergesort(x) 
    except: 
        raise ValueError("Invalid input") # catches invalid inputs
    if type(x) == dict:
        return {k:x[k] for k in keys} 
    else:
        return keys