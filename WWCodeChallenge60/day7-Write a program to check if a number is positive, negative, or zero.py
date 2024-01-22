#Write a program to check if a number is positive, negative, or zero

def check_number_sign(n):
    if n > 0:
        print(f'The number {n} is positive')
    elif n < 0:
        print(f'The number {n} is negative')
    else:
        print(f'The number {n} is zero')

# Taking input from the user
n = float(input("Enter a number: "))
check_number_sign(n)
