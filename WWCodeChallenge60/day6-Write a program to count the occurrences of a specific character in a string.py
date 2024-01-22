#Write a program to count the occurrences of a specific character in a string

def count_occurrences(input_string, target_character):
    count = 0
    for char in input_string:
        if char == target_character:
            count += 1
    return count

input_string = input("Enter a string: ")
target_character = input("Enter the character to count: ")

result = count_occurrences(input_string, target_character)
print(f"The character '{target_character}' appears {result} times in the string.")
