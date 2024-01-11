# Create a program that swaps the values of two variables.

variable1 = input()
variable2 = input()


# create a temporary variable and swap the values
swap = variable1
variable1 = variable2
variable2 = swap

print('The value of variable 1 after swapping: {}'.format(variable1))
print('The value of variable 2 after swapping: {}'.format(variable2))
