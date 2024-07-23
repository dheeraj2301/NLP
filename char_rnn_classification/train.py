
from data import *
from model import *
from config import *

from collections import Counter
from torch.utils.data import DataLoader

import torch.optim as optim




if __name__ == '__main__':


    data_path = 'data/names/*.txt'
    filenames = glob.glob(data_path)
    categories = list()
    data = list()
    labels = list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for filename in filenames:
        category = os.path.basename(filename).split('.')[0]
        categories.append(category)
        lines = read_lines(filename)
        for line in lines:
            data.append(line)
            labels.append(category)

    ### Output Tensor ###
    label_to_index_dict = labelToIndex(categories)
    label_tensor = torch.stack(outputTensor(labels,label_to_index_dict))

    ### Input Tensor ###
    data_tensor = inputTensor(data)
    dataset = NamesDataset(data_tensor, label_tensor)


    ## Class weights
    counter = Counter(labels)
    total_count = sum(counter.values())
    class_weights = [total_count / counter[label] for label in label_to_index_dict]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    ## Data Set
    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

    ## Model
    model = SimpleRNN(n_letters, hidden_size, len(label_to_index_dict), num_layers).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        for batch_data, batch_labels in data_loader:
            # Forward pass
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.argmax(dim=1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(),path)

    

