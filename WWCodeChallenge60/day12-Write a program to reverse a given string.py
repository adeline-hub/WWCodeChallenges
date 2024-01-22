#Write a program to reverse a given string

def reverse_string(input_string):
    reversed_string = input_string[::-1]
    return reversed_string

# Taking input from the user
user_input = input("Enter a string: ")
reversed_result = reverse_string(user_input)

print(f"Original String: {user_input}")
print(f"Reversed String: {reversed_result}")
