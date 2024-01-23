#Write a program to shuffle the elements of a list randomly

import random

def shuffle(lst):
    liste = random.shuffle(lst)
    return liste

# Example usage:
initial_lst = input("Input entries with space between = ").split()
result_lst = shuffle(initial_lst)

print("Shuffled List:", initial_lst)
