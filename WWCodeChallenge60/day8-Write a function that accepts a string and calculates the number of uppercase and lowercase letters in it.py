#Write a function that accepts a string and calculates the number of uppercase and lowercase letters in it


def capital_count(input_string):
    count_upper = 0
    count_lower = 0

    for character in input_string:
        if character.isupper():
            count_upper += 1
        elif character.islower():
            count_lower += 1

    print(f"Number of uppercase letters: {count_upper}")
    print(f"Number of lowercase letters: {count_lower}")

# Taking input from the user
input_string = input("Enter a string: ")
capital_count(input_string)
