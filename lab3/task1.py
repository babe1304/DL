import pandas as pd
from sklearn.metrics import confusion_matrix
import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader
from itertools import product

@dataclass
class Instance():
    def __init__(self, text, label):
        self.text = text.split(' ')
        self.label = label

class Vocab():
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    @staticmethod
    def create(dataset, max_size=10000, min_freq=1):
        text_vocab = Vocab()
        label_vocab = Vocab()

        text_vocab.stoi = {'<PAD>': 0, '<UNK>': 1}
        text_vocab.itos = {0: '<PAD>', 1: '<UNK>'}

        word_freq, label_freq = Vocab.calc_freqs(dataset)
        Vocab.build_vocab(text_vocab, word_freq, max_size, min_freq)
        Vocab.build_vocab(label_vocab, label_freq, max_size, min_freq, label=True)

        return text_vocab, label_vocab 

    @staticmethod
    def build_vocab(vocab, frequencies, max_size, min_freq, label=False):
        frequencies = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        delta = 0 if label else 2

        for i, (word, freq) in enumerate(frequencies):
            if freq >= min_freq and (len(vocab.stoi) < max_size or max_size == -1):
                vocab.stoi[word] = i + delta
                vocab.itos[i + delta] = word
            else:
                break

    def encode(self, text):
        if isinstance(text, str):
            return torch.tensor(self.stoi.get(text, 1))
        return torch.tensor([self.stoi.get(word, 1) for word in text])

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.itos[key]
        elif isinstance(key, str):
            return self.stoi[key]
        else:
            raise ValueError('key must be either int or str')

    @staticmethod 
    def calc_freqs(dataset):
        word_frequencies = {}
        label_frequencies = {}
        for instance in dataset:
            for word in instance.text:
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
            label_frequencies[instance.label] = label_frequencies.get(instance.label, 0) + 1
        return word_frequencies, label_frequencies
    
class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, path, vocab=(None, None), max_size=-1, min_freq=1):
        self.data = []
        csv = pd.read_csv(path, sep=',', header=None)
        for _, row in csv.iterrows():
            self.data.append(Instance(row[0], row[1]))

        if vocab[0] is not None:
            self.text_vocab, self.label_vocab = vocab
        else:
            self.text_vocab, self.label_vocab = Vocab.create(self.data, max_size=max_size, min_freq=min_freq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_vocab.encode(self.data[idx].text), self.label_vocab.encode(self.data[idx].label)
    
    def instance(self, idx):
        return self.data[idx]
        
class Embedding(nn.Module):
    def __init__(self, vocabular, path=None, embedding_dim=300):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(len(vocabular), embedding_dim, padding_idx=vocabular['<PAD>'])

        emb = {}
        if path is not None:
            with open(path) as f:
                for line in f:
                    word, vec = line.split(' ', 1)
                    emb[word] = torch.tensor([float(x) for x in vec.split()])
        for i, word in vocabular.itos.items():
            if word in emb:
                self.embedding.weight.data[i] = emb[word]
            elif word == '<PAD>':
                self.embedding.weight.data[i] = torch.zeros(embedding_dim)
        self.embedding.freeze = path is not None

    def forward(self, x):
        return self.embedding(x)
    
def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    max_len = max(lengths)
    padded_texts = torch.stack([F.pad(text, (0, max_len - len(text)), value=pad_index) for text in texts])
    return padded_texts, torch.tensor(labels), torch.tensor(lengths)

def train(model, data, optimizer, criterion, args):
    model.train()
    with tqdm(data, unit='batch') as t:
        for batch_num, batch in enumerate(t):
            optimizer.zero_grad()
            x, y, _ = batch
            logits = model(x).squeeze(1)
            loss = criterion(logits, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            optimizer.step()
            t.set_postfix(loss=loss.item())
    

def evaluate(model, data, criterion, args):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        cf = torch.zeros(2, 2)
        for batch_num, batch in enumerate(data):
            x, y, _ = batch
            logits = model(x).squeeze(1)
            loss = criterion(logits, y.float())
            avg_loss += loss.item()
            cf += confusion_matrix(y, logits > 0.5, labels=[0, 1])
        avg_loss /= len(data)
        acc = (cf[0, 0] + cf[1, 1]) / cf.sum()
        precision = cf[1, 1] / (cf[1, 1] + cf[0, 1])
        recall = cf[1, 1] / (cf[1, 1] + cf[1, 0])
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'Validation loss: {avg_loss}')
        print(f'Validation accuracy: {acc}')
        print(f'Validation F1: {f1}')
        print(f'Confusion matrix:\n{cf}')
        return avg_loss, acc, f1

class AveragePool(nn.Module):
    def __init__(self):
        super(AveragePool, self).__init__()

    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x
    
class RNN_Type(nn.Module):
    def __init__(self, type_, embedding, input_size, hidden_size, num_layers, bidirectional=False, dropout=0, attention=False):
        super(RNN_Type, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = embedding
        if type_ == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif type_ == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif type_ == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        else:
            raise ValueError('type must be either rnn, lstm or gru')
        
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        if attention:
            self.attention = BahdanauAttention(fc_input_size)

        # Define the fully connected layers
        self.fc_1 = nn.Linear(fc_input_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        h_n, _ = self.rnn(x)
        
        if hasattr(self, 'attention'):
            h_n = self.attention(h_n)
        else:
            h_n = h_n[:, -1, :]

        x = self.fc_1(h_n)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size//2)
        self.v = nn.Linear(hidden_size//2, 1)

    def forward(self, hidden):
        energy = torch.tanh(self.W(hidden))
        attention = F.softmax(self.v(energy), dim=1)
        context = attention * hidden
        return context.sum(dim=1)

if __name__ == '__main__':
    args = {
        'lr': 1e-4,
        'epochs': 7,
        'clip': 0.25,
        'seed': 7052020,
        'hidden_size': [300],
        'num_layers': [3],
        'dropout': [0.25],
        'bidirectional': [True],
        'vocab_size': -1,
        'batch_size': 32,
        'min_freq': 0,
    }
    attention = False
    use_embedding = True
    base = False
    models_ = ['lstm']

    train_dataset = NLPDataset('data/sst_train_raw.csv', max_size=args['vocab_size'])
    val_dataset = NLPDataset('data/sst_valid_raw.csv', vocab=(train_dataset.text_vocab, train_dataset.label_vocab))
    test_dataset = NLPDataset('data/sst_test_raw.csv', vocab=(train_dataset.text_vocab, train_dataset.label_vocab))

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    if use_embedding:
        embedding = Embedding(train_dataset.text_vocab, path='data/sst_glove_6b_300d.txt', embedding_dim=300)
    else:
        embedding = Embedding(train_dataset.text_vocab, embedding_dim=300)

    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    if base:
        print('Baseline model:')
        model = nn.Sequential(
                embedding,
                AveragePool(),
                nn.Linear(300, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, 1)
            )
        loss = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])
        for epoch in range(args['epochs']):
            train(model, train_loader, optimizer, loss, args)
            evaluate(model, val_loader, loss, args)
        print('\nTest results for baseline model:')
        base_avg_loss, base_acc, base_f1 = evaluate(model, test_loader, loss, args)

    best = {
        'f1': {'lstm': [0, None], 'gru': [0, None], 'rnn': [0, None]},
        'acc': {'lstm': [0, None], 'gru': [0, None], 'rnn': [0, None]},
        'loss': {'lstm': [float('inf'), None], 'gru': [float('inf'), None], 'rnn': [float('inf'), None]}
    }

    combinations = list(product(args['hidden_size'], args['num_layers'], args['dropout'], args['bidirectional']))
    for  i,comb in enumerate(combinations):
        args['hidden_size'], args['num_layers'], args['dropout'], args['bidirectional'] = comb
        print(f'\n {i + 1}/{len(combinations)} args: {args}')

        for type_ in models_: 
            model = RNN_Type(type_, embedding, 300, args['hidden_size'], args['num_layers'], args['bidirectional'], args['dropout'], attention)

            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=args['lr'])
            print(f'Model: {type_}')
            for epoch in range(args['epochs']):
                train(model, train_loader, optimizer, loss_fn, args)
                evaluate(model, val_loader, loss_fn, args)
            print(f'\nTest results for model {type_}:')
            loss, acc, f1 = evaluate(model, test_loader, loss_fn, args)
            if f1 > best['f1'][type_][0]:
                best['f1'][type_] = [f1, args]
            if acc > best['acc'][type_][0]:
                best['acc'][type_] = [acc, args]
            if loss < best['loss'][type_][0]:
                best['loss'][type_] = [loss, args]
            print('\n')

    if base:
        print('Baseline results:')
        print('F1:', base_f1)
        print('Accuracy:', base_acc)
        print('Loss:', base_avg_loss)
    print('\nBest results:')
    print('F1:')
    print(best['f1'])
    print('Accuracy:')
    print(best['acc'])
    print('Loss:')
    print(best['loss'])