#Write a function to count the number of vowels in a given string

string = "Coucou petit oiseau"
vowels = "aeiouAEIOU"
 
count = sum(string.count(vowel) for vowel in vowels)
print(count)
