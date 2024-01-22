#Write a program to find the maximum and minimum values in a list

def minimum_maximum(lst):
    n = int(input("Enter number of elements: "))
    my_list = []    # Initialize an empty list
    for i in range(n):
        while True:
            input_str = input("Enter element: ")
            if input_str.strip():  # Check if the input is not an empty string
                try:
                    x = int(input_str)
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
            else:
                print("Invalid input. Please enter a non-empty value.")
        my_list.append(x)
    min_val = min(my_list)    # Find the minimum and maximum values in the list
    max_val = max(my_list)
    print("List:", my_list)
    print("Minimum value in list is:", min_val)
    print("Maximum value in list is:", max_val)

# Call the function
minimum_maximum([])
