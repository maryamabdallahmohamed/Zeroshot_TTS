import string
import tensorflow as tf

class Vocab:
    def __init__(self):
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']  
        self.english_characters = list(string.ascii_lowercase + ' ')  
        self.arabic_characters = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويئءىةؤ")  
        self.characters = self.english_characters + self.arabic_characters
        self.vocab = self.special_tokens + self.characters
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    def tokenize(self, text, max_len=100, start_token=True, end_token=True):
        tokens = [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]

        if start_token:
            tokens.insert(0, self.char2idx['<SOS>'])
        if end_token:
            tokens.append(self.char2idx['<EOS>'])

        if max_len is not None:
            tokens = tokens[:max_len]  
            tokens += [self.char2idx['<PAD>']] * (max_len - len(tokens))  

        return tokens
    def decode(self, sentence):
        out = ''
        for token in sentence:
            if isinstance(token, tf.Tensor):
                token = token.numpy().item()
            char = self.idx2char[token]
            if char == '<EOS>':
                return out
            if not (char in self.special_tokens):
                out += char
        return out
    

# decoded=test.decode(encoded)
# print(decoded)
# special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']  
# english_characters = list(string.ascii_lowercase + ' ')  
# arabic_characters = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويئءىةؤ")  

# characters = english_characters + arabic_characters
# vocab = special_tokens + characters 

# char2idx = {char: idx for idx, char in enumerate(vocab)}
# idx2char = {idx: char for idx, char in enumerate(vocab)}

# vocab_size = len(vocab)
# print(f"Vocabulary size: {vocab_size}")

# print(vocab)

# def tokenize_text(text, char2idx, max_len=100, start_token=True, end_token=True):
#     tokens = [char2idx.get(char, char2idx['<UNK>']) for char in text]

#     if start_token:
#         tokens.insert(0, char2idx['<SOS>'])
#     if end_token:
#         tokens.append(char2idx['<EOS>'])

#     if max_len is not None:
#         tokens = tokens[:max_len]  
#         tokens += [char2idx['<PAD>']] * (max_len - len(tokens))  

#     return tokens



# def TextDecoder(sentence):
#     out = ''
#     for token in sentence:
#         if isinstance(token, tf.Tensor):
#             token = token.numpy().item()
#         char = idx2char[token]
#         if char == '<EOS>':
#             return out
#         if not (char in special_tokens):
#             out += char
#     return out
