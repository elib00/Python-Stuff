import json
from nltk_utils import tokenize, stem_word, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from training_model import NeuralNetwork

class ChatDataSet(Dataset):
    def __init__(self, x_train, y_train):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = torch.LongTensor(y_train)
            
    #dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.n_samples

def main():
    with open("intents.json", "r") as f:
        intents = json.load(f)
        
    word_collection = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
            
        for pattern in intent["patterns"]:
            word = tokenize(pattern) #arrray of words
            word_collection.extend(word)
            xy.append((word, tag))

    ignore = ['?', '!', ',', '.', '&']
    word_collection = [stem_word(word) for word in word_collection if word not in ignore] 
    word_collection = sorted(set(word_collection))
    tags = sorted(set(tags))

    x_train = []
    y_train = []
        
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, word_collection)
        x_train.append(bag)  
            
        label = tags.index(tag)
        y_train.append(label)
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)

            
    #Hyperparameter
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(word_collection)
    learning_rate = 0.001
    num_epochs = 1000
        
    dataset = ChatDataSet(x_train, y_train)
    train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
        
        #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
                
            #forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
                
            #backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1} / {num_epochs}, loss = {loss.item():.4f}")
                
    print(f"final loss, loss = {loss.item():.4f}")  
    
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "word_collection": word_collection, 
        "tags": tags
    }    
    
    FILE = "data.pth"
    torch.save(data, FILE)
    
    print(f"Training complete. File saved to {FILE}")
        
        
if __name__ == "__main__":
    main()