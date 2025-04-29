import re
from num2words import num2words

def sentence_preprocessing(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("\n", " ")
    sentence = re.sub(r'[إأآ]', 'ا', sentence)
    sentence = re.sub(r'[^a-zA-Zء-ي\s\d]', '', sentence)
    pattern = r'[\u0617-\u061A\u064B-\u065F]'
    sentence = re.sub(pattern, '', sentence)
    pattern = r'([a-zA-Z])([ء-ي])|([ء-ي])([a-zA-Z])'
    sentence = re.sub(pattern, r'\1\3 \2\4', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang='ar'), sentence)
    return sentence