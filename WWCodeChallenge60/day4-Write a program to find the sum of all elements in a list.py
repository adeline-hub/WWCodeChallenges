#Write a program to find the sum of all elements in a list.

def count_elements(list):
    count = 0
    list = []
    n = int (input ("Enter number of elements: ")) 
    for i in range (n):
        x = int(input())
        list.append(x)
        count = sum(list)
    print("List:", list, count)
    print("Sum of elements is:", count)

count_elements(list)
