#Write a function that counts the frequency of each word in a sentence.

def count_word_frequency(sentence):
    # Split the sentence into words
    words = sentence.split()

    # Create an empty dictionary to store word frequencies
    word_frequency = {}

    # Count the frequency of each word
    for word in words:
        # Remove punctuation and convert to lowercase for better matching
        word = word.strip('.,!?').lower()

        # Update the word frequency in the dictionary
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1

    return word_frequency

# Example usage:
sentence = input() #"This is a sample sentence. This sentence contains some sample words."
result = count_word_frequency(sentence)
print("Word frequency in the sentence:")
for word, frequency in result.items():
    print(f"{word}: {frequency}")
