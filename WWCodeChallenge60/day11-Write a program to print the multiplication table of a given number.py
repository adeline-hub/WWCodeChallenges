#Write a program to print the multiplication table of a given number

def calculate(num):
    for i in range(1, 11):
        print(f"{num} x {i} = {num * i}")


num = int(input('Enter your number: '))
calculate(num)
