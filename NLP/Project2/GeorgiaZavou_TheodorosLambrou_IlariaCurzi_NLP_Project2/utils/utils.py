from collections import defaultdict
import torch
import torch.nn as nn
import pandas as pd




class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTM_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return logits


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix, max_len=50):
        self.data = []
        self.max_len = max_len
        for words, tags in sentences:
            word_ids = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in words]
            tag_ids = [tag_to_ix[t] for t in tags]
            if len(word_ids) < max_len:
                pad_len = max_len - len(word_ids)
                word_ids += [word_to_ix["<PAD>"]] * pad_len
                tag_ids += [-100] * pad_len  # Ignored by loss function
            else:
                word_ids = word_ids[:max_len]
                tag_ids = tag_ids[:max_len]
            self.data.append((torch.tensor(word_ids), torch.tensor(tag_ids)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


import pandas as pd
from collections import defaultdict, Counter

# Load data 

def load_data(train_path, test_path, tiny_test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    tiny_test_df = pd.read_csv(tiny_test_path)
    return train_df, test_df, tiny_test_df


# Group sentences by sentence_id


def group_sentences(df):
    sentences = defaultdict(list)
    for row in df.itertuples(index=False):
        sentences[row.sentence_id].append((row.words, row.tags))
    
    data = []
    for sid in sorted(sentences.keys()):
        words, tags = zip(*sentences[sid])
        data.append((list(words), list(tags)))
    return data


# Build vocabularies for DL models

def build_vocab(dataset, min_freq=1):
    word_counter = Counter()
    tag_counter = Counter()
    
    for words, tags in dataset:
        word_counter.update([str(w).lower() for w in words if pd.notnull(w)])
        tag_counter.update(tags)
    
    vocab_words = [w for w, c in word_counter.items() if c >= min_freq]
    vocab_tags = list(tag_counter.keys())
    
    word2idx = {w: i+2 for i, w in enumerate(vocab_words)}
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    
    tag2idx = {t: i for i, t in enumerate(vocab_tags)}
    return word2idx, tag2idx
