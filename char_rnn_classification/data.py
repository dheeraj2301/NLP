import os
import glob
import string
import unicodedata
import torch
import string
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read lines from a file and convert to ASCII
def read_lines(filename):
    with open(filename, encoding='utf-8') as file:
        lines = file.read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

# Constants
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters> tensor
def line_to_tensor(line):
    tensor = torch.zeros(len(line),n_letters)
    for li, letter in enumerate(line):
        tensor[li][letter_to_index(letter)] = 1
    return tensor
def extractCategory(filename):
    return os.path.basename(filename).split('.')[0]

def labelToIndex(categories):
 return {category:i for i ,category in enumerate(categories)}

def labelToTensor(label,n_labels):
    tensor = torch.zeros(n_labels)
    tensor[label] = 1
    return tensor

def outputTensor(outputs,output_index_dict):
    n_labels = len(output_index_dict)
    output_tensor = [labelToTensor(output_index_dict[output],n_labels) for output in outputs]
    return output_tensor

def inputTensor(input):
    tensors = []
    for i,name in enumerate(input):
        name_tensor = line_to_tensor(name)
        tensors.append(name_tensor)

    input_tensor = pad_sequence(tensors,batch_first= True)

    return input_tensor



class NamesDataset(Dataset):
    def __init__(self,data_tensor,label_tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, index):
        return self.data_tensor[index],self.label_tensor[index]
    
