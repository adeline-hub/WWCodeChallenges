#Write a function to calculate the factorial of a number.

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

# Driver Code
num = int(input("Enter a number: "))
print("Factorial of", num, "is", factorial(num))
