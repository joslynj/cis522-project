import argparse
import json
import logging
import os
import sagemaker_containers
import sys
import csv
import ast

import pandas as pd
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

import torch 
import torchtext
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class NoteEncoder(nn.Module):
  def __init__(self, embeddings, hidden_size, embedding_length):
    super(NoteEncoder, self).__init__()

    # Specifications
    self.hidden_size = hidden_size
    self.embedding_length = embedding_length

    # Embedding Layer
    self.embeddings = embeddings

    # Note Encoding Layer
    self.recurrent = nn.LSTM(self.embedding_length, self.hidden_size)

  def forward(self, note, note_lengths):
    embedded_note = self.embeddings(note)
    packed_note = nn.utils.rnn.pack_padded_sequence(embedded_note, note_lengths, enforce_sorted=False)
    outputs, (hidden, _) = self.recurrent(embedded_note)
    return hidden.squeeze(0)

"""A model to compute attention scores."""

class Attention(nn.Module):
  def __init__(self, hidden_size):
    super(Attention, self).__init__()
    self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 1)
    self.relu = nn.ReLU();
    self.softmax = nn.Softmax(dim = 0)

  def forward(self, hidden, encoder_outputs):
    repeated_hidden = hidden.repeat(encoder_outputs.size()[0], 1, 1)
    concated_input = torch.cat((encoder_outputs, repeated_hidden), dim = 2)
    return self.softmax(self.fc2(self.relu(self.fc1(concated_input))))

"""The note order model takes in samples with dimensions (batch size, sample length, note length) and lengths with dimensions (batch size, max sample length). It instantiates an encoder and an attention layer. First, it encodes each note for a sample using the encoder. Then, it applies an lstm on the note. The output is inputted into the attention model and finally into a fully connected layer of size - number of labels."""

class NoteOrderDiagnosis(nn.Module):
  def __init__(self, output_size, hidden_size_1, hidden_size_2, vocab_size, embedding_length, word_embeddings, device):
    super(NoteOrderDiagnosis, self).__init__()

    self.device = device

    self.hidden_size_1 = hidden_size_1
    self.hidden_size_2 = hidden_size_2
    self.output_size = output_size

    self.embeddings = nn.Embedding.from_pretrained(word_embeddings)


    self.recurrent =  nn.LSTM(hidden_size_1, hidden_size_2)
    self.attention = Attention(hidden_size_2)
    self.fc = nn.Linear(2*hidden_size_2, output_size)

    self.encoder = NoteEncoder(self.embeddings, hidden_size_1, embedding_length)

    
  def forward(self, samples, lengths):
    sample_lengths = [np.count_nonzero(note_lengths) for note_lengths in lengths]
    final = torch.zeros(len(samples[0]), len(samples), self.hidden_size_1).to(self.device)

    # Encoding Phase
    for i in range(len(samples)):
      sample = samples[i].view(samples[i].size()[1], samples[i].size()[0])
      encoded =  self.encoder(sample[:, 0:sample_lengths[i]], lengths[i][lengths[i] != 0])
      final[0:sample_lengths[i], i, :] = encoded

    # Recurrence By Note Order
    final = nn.utils.rnn.pack_padded_sequence(final, sample_lengths, enforce_sorted=False)
    output, (hidden, _) = self.recurrent(final)
    (output, _) = nn.utils.rnn.pad_packed_sequence(output)

    # Attention Mechanism
    batch_len = output.size()[1]
    attention_scores = self.attention(hidden, output)
    context_vec = torch.bmm(output.view(batch_len, self.hidden_size_2, -1), attention_scores.view(batch_len, -1, 1))
    context_concated = torch.cat((hidden, context_vec.view(1, batch_len, self.hidden_size_2)), dim = 2)
    
    return self.fc(context_concated).squeeze(0)





def string_to_list(l):
  return ast.literal_eval(l)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _get_iterators(batch_size, training_dir, valid_dir, is_distributed, device, **kwargs):
    logger.info("Get train data loader")

    csv.field_size_limit(sys.maxsize)

    NOTE = torchtext.data.Field(sequential=True, lower=True)
    SAMPLE = torchtext.data.NestedField(nesting_field=NOTE, tokenize=string_to_list, include_lengths=True, use_vocab=True)
    ICD9 = torchtext.data.Field(sequential=True, tokenize=string_to_list, use_vocab=False)

    """Reading in our CSV files into the NestedField.  Schema when loading in CSV is COL_NUM, Unnamed: 0, HADM_ID, NOTES_SORTED, ONE_HOT."""

    datafields = [(" ", None), \
                  (" ", None), \
                  ("samples", SAMPLE), \
                  ("icd9", ICD9)]

    training_data=torchtext.data.TabularDataset(path = training_dir,\
                                      format = 'csv',\
                                      fields = datafields,\
                                      skip_header = True)

    testing_data=torchtext.data.TabularDataset(path = valid_dir,\
                                        format = 'csv',\
                                        fields = datafields,\
                                        skip_header = True)
      
    SAMPLE.build_vocab(training_data, testing_data, min_freq = 3, vectors=torchtext.vocab.GloVe(name='6B', dim=300))

    # Define the train iterator
    train_iterator = torchtext.data.BucketIterator(
        training_data, 
        batch_size = batch_size,
        sort_key = lambda x: len(x.samples),
        sort_within_batch = True,
        repeat=False, 
        shuffle=True,
        device = device)

    # Define the test iterator
    test_iterator = torchtext.data.BucketIterator(
        testing_data, 
        batch_size = batch_size,
        sort=False,
        sort_key = lambda x: len(x.samples),
        sort_within_batch = False,
        repeat=False, 
        shuffle=False,
        device = device)
    
    return train_iterator, test_iterator, len(SAMPLE.vocab), SAMPLE.vocab.vectors


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def evaluate_classifier(model, test_iterator, device, loss_function):
  print("~~~Started Evaluation~~~")
  total = 0
  total_loss = 0
  y_pred = None
  y_true = None
  with torch.no_grad():
    for batch in test_iterator:
      samples, _, lengths = batch.samples
      samples = samples.to(device)
      lengths = lengths.to(device)


      icd9 = batch.icd9.permute(1, 0).float()     
      batch_size = len(icd9)

      output = model(samples, lengths)
      loss = loss_function(output, icd9)

      pred = (torch.sigmoid(output).data > 0.5).int()
      if y_pred is None:
        y_pred = pred.cpu().numpy()
      else:
        y_pred = np.concatenate((y_pred, pred.cpu().numpy()))
      if y_true is None:
        y_true = icd9.int().cpu().numpy()
      else:
        y_true = np.concatenate((y_true, icd9.int().cpu().numpy()))
      total += batch_size
      total_loss += loss.item()
      print("   # Processed:", total)

  print("VALID RECALL SCORE", recall_score(y_true, y_pred, average='micro'))
  print("VALID PRECISION SCORE", precision_score(y_true, y_pred, average='micro'))
  acc = np.sum(y_pred == y_true) / (total * model.output_size)
  print("Test statistics: Acc: %s Loss: %s"%(acc, total_loss / total))
  return acc, total_loss / total, y_pred

def train_classifier(model, dataset_iterator, test_iterator, loss_function, optimizer, num_epochs, device, is_distributed, use_cuda):
  print("~~~Started Training~~~")
  train_loss = []
  train_acc = []
  model.train()
  for epoch in range(num_epochs):
    print("Epoch", epoch + 1)
    total = 0
    total_loss = 0
    total_f1 = 0
    y_pred = None
    y_true = None
    for batch in dataset_iterator:
      samples, _, lengths = batch.samples

      samples = samples.to(device)
      lengths = lengths.to(device)

      icd9 = batch.icd9.permute(1, 0).float()

      batch_size = len(icd9)

      optimizer.zero_grad()
      output = model(samples, lengths)
      loss = loss_function(output, icd9)
      loss.backward() 
      if is_distributed and not use_cuda:
        # average gradients manually for multi-machine cpu case only
        _average_gradients(model)
      optimizer.step()

      pred = (torch.sigmoid(output).data > 0.5).int()
      if y_pred is None:
        y_pred = pred.cpu().numpy()
      else:
        y_pred = np.concatenate((y_pred, pred.cpu().numpy()))
      if y_true is None:
        y_true = icd9.int().cpu().numpy()
      else:
        y_true = np.concatenate((y_true, icd9.int().cpu().numpy()))

      total_loss += loss.item()
      total += batch_size
      print("   # Processed this Epoch:", total)

    acc = np.sum(y_pred == y_true) / (total * model.output_size)
    for i in range(0, len(y_pred)):
      f1 = f1_score(y_true[i], y_pred[i])
      total_f1 += f1
    print("Epoch: %s Acc: %s Loss: %s"%(epoch + 1, acc, total_loss / total))
    print("TRAIN RECALL SCORE", recall_score(y_true, y_pred, average='micro'))
    print("TRAIN PRECISION SCORE", precision_score(y_true, y_pred, average='micro'))
    print("F1:", total_f1)
    
    train_loss.append(total_loss / total)
    train_acc.append(acc)
    evaluate_classifier(model, test_iterator, device, loss_function)
  return train_loss, train_acc

def train(args):
  is_distributed = len(args.hosts) > 1 and args.backend is not None
  logger.debug("Distributed training - {}".format(is_distributed))
  use_cuda = args.num_gpus > 0
  logger.debug("Number of gpus available - {}".format(args.num_gpus))
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  device = torch.device("cuda" if use_cuda else "cpu")

  if is_distributed:
      # Initialize the distributed environment.
      world_size = len(args.hosts)
      os.environ['WORLD_SIZE'] = str(world_size)
      host_rank = args.hosts.index(args.current_host)
      os.environ['RANK'] = str(host_rank)
      dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
      logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
          args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
          dist.get_rank(), args.num_gpus))

  # set the seed for generating random numbers
  torch.manual_seed(args.seed)
  if use_cuda:
      torch.cuda.manual_seed(args.seed)

  train_iterator, test_iterator, vocab_size, word_embeddings = _get_iterators(args.batch_size, args.train_dir, args.valid_dir, is_distributed, device, **kwargs)

  train_df = pd.read_csv(args.train_dir)
  agg = train_df['one_hot'].apply(string_to_list)
  output = np.zeros((args.output_size))

  for i in range(len(agg)):
    output = output + agg[i]

  totals = np.ones((args.output_size)) * len(agg)
  negatives = totals - output

  weights = negatives/output

  hidden_size_1 = args.hidden_size_1
  hidden_size_2 = args.hidden_size_1
  lr = args.lr
  embedding_length = 300
  output_size = args.output_size
  num_epochs = args.epochs
  loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights))

  model = NoteOrderDiagnosis(output_size, hidden_size_1, hidden_size_2, vocab_size, embedding_length, word_embeddings, device)
  model.to(device)

  if is_distributed and use_cuda:
      # multi-machine multi-gpu case
      model = torch.nn.parallel.DistributedDataParallel(model)
  else:
      # single-machine multi-gpu case or single-machine or multi-machine cpu case
      model = torch.nn.DataParallel(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  train_loss, train_acc = train_classifier(model, train_iterator, test_iterator, loss_function, optimizer, num_epochs, device, is_distributed, use_cuda)
  print('----------Train Loss----------')
  print(train_loss)

  print('----------Train Accuracy----------')
  print(train_acc)
  save_model(model, args.model_dir)



def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)



def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--hidden-size-1', type=int, default=256, metavar='HS1',
                        help='hidden size 1 (default: 256)')
    parser.add_argument('--hidden-size-2', type=int, default=128, metavar='HS2',
                        help='hidden size 2 (default: 128)')
    parser.add_argument('--output-size', type=int, default=1000, metavar='OS',
                        help='output size (default: 1000)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TESTING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())