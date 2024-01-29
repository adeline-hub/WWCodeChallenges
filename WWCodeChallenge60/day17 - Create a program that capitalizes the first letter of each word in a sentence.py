#Create a program that capitalizes the first letter of each word in a sentence

def capitalize_first_letter(sentence):
    # Split the sentence into words
    words = sentence.split()

    # Capitalize the first letter of each word and join them back into a sentence
    capitalized_sentence = ' '.join([word.capitalize() for word in words])

    return capitalized_sentence

# Example usage:
input_sentence = "this is a sample sentence."
capitalized_result = capitalize_first_letter(input_sentence)

print(f"Original sentence: {input_sentence}")
print(f"Capitalized sentence: {capitalized_result}")
