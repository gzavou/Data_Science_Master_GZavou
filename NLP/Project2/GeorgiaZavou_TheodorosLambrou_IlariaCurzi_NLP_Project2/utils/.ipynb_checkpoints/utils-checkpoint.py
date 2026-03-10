from collections import defaultdict
import torch
import torch.nn as nn
import pandas as pd


def group_sentences(df):
    """Group word-tag pairs by sentence ID."""
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["sentence_id"]].append((row["words"], row["tags"]))
    return list(grouped.values())


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



def clean_sentences(sentences): #not used anymore
    """Replace non-string or NaN words and tags with <UNK> and 'O' respectively."""
    cleaned = []
    for sentence in sentences:
        cleaned_sentence = [
            (w if isinstance(w, str) and pd.notna(w) else "<UNK>",
             t if isinstance(t, str) and pd.notna(t) else "O")
            for w, t in sentence
        ]
        cleaned.append(cleaned_sentence)
    return cleaned

