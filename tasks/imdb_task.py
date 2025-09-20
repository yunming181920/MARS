import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tasks.task import Task
from torch.utils.data import Subset
import numpy as np
import random
max_features = 10000  # 词汇表大小
maxlen = 500  # 每个评论的最大长度
embedding_dim = 32  # 嵌入层的维度
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, hn = self.rnn(x)
        hn = hn.squeeze(0)
        out = self.fc(hn)
        out = self.sigmoid(out)
        return out
class IMDBTask(Task):
    def load_data(self):
        self.load_imdb_data()
        split = min(self.params.fl_total_participants / 20, 1)

        random_index = np.random.choice(len(self.train_dataset), len(self.train_dataset), replace=False)
        self.train_dataset = Subset(self.train_dataset, random_index)
        all_range = list(range(int(len(self.train_dataset) * split)))
        random.shuffle(all_range)
        train_loaders = [self.get_train_old(all_range, pos)
                         for pos in
                         range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
    def load_imdb_data(self):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


        x_train = pad_sequences(x_train, maxlen=maxlen)
        x_test = pad_sequences(x_test, maxlen=maxlen)


        x_train = torch.tensor(x_train, dtype=torch.long)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.float32)


        self.train_dataset = TensorDataset(x_train, y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_dataset = TensorDataset(x_test, y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
    def build_model(self) -> None:
        return RNNModel(max_features, embedding_dim, 32, 1)