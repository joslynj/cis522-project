import torch
from transformers import *

#import pretrained model and output all hidden states from model

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
config = BertConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", config=config)
model.eval()

#do not have actual notes but tokenize an example (get word indices)...

example_note = "patient has severe case of epithelial trauma in lungs"
marked_note = "[CLS] " + example_note + " [SEP]"

tokenized_note = tokenizer.tokenize(marked_note)
indexed_note = tokenizer.convert_tokens_to_ids(tokenized_note)
segments_ids = [1] * len(indexed_note)

tokenized_note = tokenizer.tokenize(marked_note)

# Convert inputs to PyTorch tensors

tokens_tensor = torch.tensor([indexed_note])
segments_tensors = torch.tensor([segments_ids])

# Acquire hidden states

with torch.no_grad():
    last_hidden_state, pooler_output, hidden_states  = model(tokens_tensor)

print(torch.eq(hidden_states[-1], last_hidden_state))
print(hidden_states[-2].shape)

sent_embedding = torch.mean(hidden_states[-2][0], dim = 0)

# Check out cosine similarity between embeddings..

example_note_2 = "long history of frequent respiratory infections, shortness of breath, or wheezing"
example_note_3 = "Patient is showing nausea, persistent diarrhea, and vomiting,"

marked_note_2 = "[CLS] " + example_note_2 + " [SEP]"
marked_note_3 = "[CLS] " + example_note_3 + " [SEP]"

tokenized_note_2 = tokenizer.tokenize(marked_note_2)
tokenized_note_3 = tokenizer.tokenize(marked_note_3)

indexed_note_2 = tokenizer.convert_tokens_to_ids(tokenized_note_2)
indexed_note_3 = tokenizer.convert_tokens_to_ids(tokenized_note_3)

tokens_tensor_2 = torch.tensor([indexed_note_2])
tokens_tensor_3 = torch.tensor([indexed_note_3])

with torch.no_grad():
    _, _, hidden_states_2  = model(tokens_tensor_2)
    _, _, hidden_states_3 = model(tokens_tensor_3)

sent_embedding_2 = torch.mean(hidden_states_2[-2][0], dim = 0)
sent_embedding_3 = torch.mean(hidden_states_3[-2][0], dim = 0)

from scipy.spatial.distance import cosine

close_sent = 1 - cosine(sent_embedding, sent_embedding_2)

diff_sent = 1 - cosine(sent_embedding, sent_embedding_3)

print(close_sent)
print(diff_sent)