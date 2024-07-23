from model import *
from data import *
import sys
from config import *
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleRNN(n_letters, hidden_size, 18, num_layers=num_layers).to(device)
model.load_state_dict(torch.load(path))
model.eval()

label_to_index_dict = {'Arabic': 0,
 'Chinese': 1,
 'Czech': 2,
 'Dutch': 3,
 'English': 4,
 'French': 5,
 'German': 6,
 'Greek': 7,
 'Irish': 8,
 'Italian': 9,
 'Japanese': 10,
 'Korean': 11,
 'Polish': 12,
 'Portuguese': 13,
 'Russian': 14,
 'Scottish': 15,
 'Spanish': 16,
 'Vietnamese': 17}

def preprocess_input(name):
    tensor = line_to_tensor(unicode_to_ascii(name))
    if tensor.size(0) < max_length:
        pad_size = max_length - tensor.size(0)
        tensor = F.pad(tensor, (0, 0, 0, pad_size))
    tensor = tensor.unsqueeze(0)
    # tensor = tensor.unsqueeze(0)
    return tensor.to(device)



def predict(name, top_n=3):
    model.eval()
    with torch.no_grad():
        input_tensor = preprocess_input(name)
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        top_n_values, top_n_indices = torch.topk(probabilities, top_n, dim=1)
        top_n_indices = top_n_indices.cpu().numpy().flatten()
        top_n_labels = [list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(idx)] for idx in top_n_indices]
    return top_n_labels, top_n_values.cpu().numpy().flatten()

def prediction(input,n):
    
    top_n_predictions, top_n_scores = predict(input, n)
    print("Top predicted Categories are: ")
    for label, score in zip(top_n_predictions, top_n_scores):
        print(f"{label} -> {score*100:.1f}%")

if __name__ == '__main__':
    prediction(sys.argv[1],3)