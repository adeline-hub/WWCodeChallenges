#Write a function to count the number of vowels in a given string

def count_vowels(sentence):
    vowels = "aeiouAEIOU"
    count = 0
    for character in sentence:
        if character in vowels:
            count +=1
    return count


sentence = input("Enter your sentence: ")
vowel_count = count_vowels(sentence)
print("In the sentence, vowels count for:", vowel_count)
