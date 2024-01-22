#Write a program to remove duplicates from a list

def remove_duplicates(input_list):
    unique_list = list(set(input_list)) # sets do not allow duplicate elements
    return unique_list

# Example usage:
original_list = input("Input entries with space between = ")
original_list = original_list.split(" ")
result_list = remove_duplicates(original_list)

print("Original List:", original_list)
print("List with Duplicates Removed:", result_list)
