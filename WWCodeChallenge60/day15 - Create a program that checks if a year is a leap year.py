#Create a program that checks if a year is a leap year

def is_leap_year(year):
    # Leap year is divisible by 4
    if (year % 4) == 0:
        # If divisible by 100, it should also be divisible by 400
        if (year % 100) == 0:
            if (year % 400) == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

# Get user input for the year
year = int(input("Enter a year: "))

# Check if the entered year is a leap year or not
if is_leap_year(year):
    print(f"{year} is a leap year.")
else:
    print(f"{year} is not a leap year.")
