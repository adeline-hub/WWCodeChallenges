#Write a program to check if a number is even or odd

def odd_or_not(num):
    if num % 2 != 0:
        print(f'The number {num} is odd')
    else:
        print(f'The number {num} is even')

# Taking input from the user
num = int(input('Enter your number: '))
odd_or_not(num)
