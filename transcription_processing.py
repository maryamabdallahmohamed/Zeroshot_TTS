import re
from num2words import num2words

def transcription_preprocessing(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("\n", " ")

    # Replace the variants of 2lf
    sentence = re.sub(r'[إأآ]', 'ا', sentence)

    # Remove the punctuation and special characters
    sentence = re.sub(r'[^a-zA-Zء-ي\s\d]', '', sentence)  # Keep only Arabic, English letters, digits, and spaces

    pattern = r'[\u0617-\u061A\u064B-\u065F]'

    # Remove all the arabic special characters (تشكيل)
    sentence = re.sub(pattern, '', sentence)

    # Pattern to insert a space between Arabic and English (or vice versa)
    pattern = r'([a-zA-Z])([ء-ي])|([ء-ي])([a-zA-Z])'

    # Replace matches with the same characters separated by a space
    sentence = re.sub(pattern, r'\1\3 \2\4', sentence)
    # Replace multiple spaces with a single space
    sentence = re.sub(r'\s+', ' ', sentence)

    # Regular expression to find numbers in the sentence and replace them with words
    sentence = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang='ar'), sentence)
    return sentence