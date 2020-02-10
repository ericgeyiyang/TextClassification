import os
import io
import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator


class CNN_1d(nn.Module):
    def __init__(self, embed_dim, num_classes, num_kernels, kernel_size_list, dropout_prob, vocab):
        super(CNN_1d, self).__init__()
        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_kernels, (k, embed_dim)) for k in kernel_size_list])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(len(kernel_size_list)*num_kernels, num_classes, )


    def forward(self, x):

        x = self.embed(x)  # (L, N, embed_dim)
        x = x.permute(1, 0 , 2) # (N, L, embed_dim)

        x = x.unsqueeze(1)  # (N, 1, L, emebed_dim)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, num_kernels, L)]*len(Ks)

        x = [F.max_pool1d(i, kernel_size = i.size(2)).squeeze(2) for i in x]  # [(N, num_kernels)]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*num_kernels)
        logit = self.fc(x)  # (N, C)
        return logit

def train_epoch(model, dataloader, optimizer, criterion, bs):
    model.train()
    # Iterate over batches
    total_loss, corrects, step = 0, 0, 0
    
    for batch in dataloader:
        step += 1
        optimizer.zero_grad()
        # Various inputs
        vecs, labels = batch #[L, N], [N]
        labels -= 1
        logits = model(vecs)
        loss = criterion(logits, labels)
        loss.backward()
        total_loss += loss.detach().data
        # Optimizer step
        optimizer.step()
        t = (torch.max(logits, 1, keepdim=True)[1].view(labels.size()).data == labels.data).sum()
        corrects += t.cpu().numpy()
        if step % 1000 == 0:
            accuracy = 100 * corrects / (bs * step)
            print(f'Batch[{step:d}] avg_loss: {(total_loss / step):.2f}  acc: {accuracy:.4f}%')


def evaluate(model, dataloader, criterion):
    model.eval()
    corrects, total_loss = 0, 0
    step = 0
    for batch in dataloader:
        step += 1
        vecs, labels = batch
        labels -= 1
        # padding vecs 
        with torch.no_grad():
            # Get log probs
            logits = model(vecs)
            loss = criterion(logits, labels)
            total_loss += loss.detach().data
            t = (torch.max(logits, 1, keepdim=True)[1].view(labels.size()).data == labels.data).sum()
            corrects += t.cpu().numpy()
    return total_loss / step, corrects

def predict(model, dataloader):
    model.eval()
    step = 0
    pred = []
    # pdb.set_trace()
    for batch in dataloader:
        step += 1
        vecs, _ = batch
        with torch.no_grad():
            logits = model(vecs)
            pred += list((torch.argmax(logits.data, 1, keepdim=True)).cpu().numpy().reshape(-1))
    return pred

def load_index2label(x = 'topicclass/label_map.txt'):
    d = {}
    with open(x) as f:
        for line in f:
            parsed = line.split('|')
            d[int(parsed[1]) - 1] = parsed[0]
    return d

parser = argparse.ArgumentParser("Train an test classifier model")
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--test-only", action="store_true")
parser.add_argument("--model-file", type=str, default="model.pt")

args = parser.parse_args()

BATCH_SIZE= 64
numEpochs = 20

num_kernels = 50
kernel_size_list = [3,4,5]

max_norm = 3.
dropout_prob = .5

embed_dim = 300
num_classes = 16
learningRate = 1e-3
lr_decay = 0.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# code reference from https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)

LABEL = Field(sequential=False, use_vocab=False)

tv_datafields = [("label", LABEL),
                 ("topic", TEXT)]
train, valid = TabularDataset.splits(
               path="topicclass", # the root directory where the data lies
               train='topicclass_train.csv', validation="topicclass_valid.csv",
               format='csv',
               skip_header=True,
               fields=tv_datafields)

tst_datafields = [("topic", TEXT)]
test = TabularDataset(
           path="topicclass/topicclass_valid.csv", # the file path
           format='csv',
           skip_header=True, 
           fields=tv_datafields)

TEXT.build_vocab(train)

vocab = TEXT.vocab
train_iter, val_iter = BucketIterator.splits(
 (train, valid), # we pass in the datasets we want the iterator to draw data from
 batch_sizes=(BATCH_SIZE, BATCH_SIZE),
 device=device, # if you want to use the GPU, specify the GPU number here
 sort_key=lambda x: len(x.topic), # the BucketIterator needs to be told what function it should use to group the data.
 sort_within_batch=False,
 repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)
test_iter = Iterator(test, batch_size=BATCH_SIZE, device=device, shuffle=False, sort=False, sort_within_batch=False, repeat=False)

class BatchWrapper:
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x 

      def __iter__(self):
            for batch in self.dl:
                  x = getattr(batch, self.x_var) # we assume only one input in this wrapper

                  if self.y_vars: # we will concatenate y into a single tensor
                        y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).long()
                        y = y.squeeze(1)
                  else:
                        y = torch.zeros((1))

                  yield (x, y)

      def __len__(self):
            return len(self.dl)

train_dl = BatchWrapper(train_iter, "topic", ['label'])
valid_dl = BatchWrapper(val_iter, "topic", ['label'])
test_dl = BatchWrapper(test_iter, "topic", ['label'])
model = CNN_1d(embed_dim, num_classes, num_kernels, kernel_size_list, dropout_prob, vocab)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4], gamma=0.1)
if os.path.isfile(args.model_file):
    print('loading models............')
    model.load_state_dict(torch.load(args.model_file))
model = model.to(device)
index2label = load_index2label()


if args.validate_only:
    avg_loss, corrects = evaluate(model, valid_dl, criterion)
    acc = 100 * corrects / (len(valid))
    print(f"Validation loss: {avg_loss:.2f}, acc: {acc:.4f}%")
elif args.test_only:
    pred = predict(model, test_dl)
    with open("result.txt", 'w') as f:
        for x in pred:
            f.write(index2label[x])
            f.write("\n")
    print(f"test completed")
else:
    best_acc = 0
    for epoch in range(1, numEpochs+1):
        print(f"----- Epoch {epoch} -----", flush=True)
        # Train for one epoch
        train_epoch(model, train_dl, optimizer, criterion, BATCH_SIZE)
        # Check dev ppl
        avg_loss, corrects = evaluate(model, valid_dl, criterion)
        acc = 100 * corrects / (len(valid))
        print(f"Validation loss: {avg_loss:.2f}, acc: {acc:.4f}%")
        # Early stopping maybe
        if acc > best_acc:
            best_acc = acc
            print(f"Saving new best model (epoch {epoch} acc {acc})")
            torch.save(model.state_dict(), args.model_file)
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay




