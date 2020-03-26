#!pip install -U scispacy
#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gzing
#!pip install -U sentence-transformers

import string
import numpy as np
import en_core_sci_md
from sentence_transformers import SentenceTransformer

#lung passage
passage1 = "He has a condition in which the alveoli at the end of the smallest air passages (bronchioles) \
of the lungs are destroyed as a result of damaging exposure to cigarette smoke and other irritating gases and particulate matter. \
Symptoms include breathing difficulty, cough, mucus (sputum) production and wheezing."

#lung passage
passage2 = 'patient is showing interstitial lung disease. Injury to his lungs has triggerd an abnormal healing response. \
Ordinarily, body generates just the right amount of tissue to repair damage. \
But in his case, the repair process goes awry and the tissue around the air sacs (alveoli) becomes scarred and thickened. \
This makes it more difficult for oxygen to pass into his bloodstream.'

#brain passage
passage3 = "Symptoms include mental decline, difficulty thinking and understanding, \
confusion in the evening hours, delusion, disorientation, forgetfulness, making things up, \
mental confusion, difficulty concentrating, inability to create new memories, \
inability to do simple math, or inability to recognize common things. Also, expressing inability to combine muscle \
movements, jumbled speech, or loss of appetite"

#brain passage
passage4 = "Patient has stiff muscles, difficulty standing, difficulty walking, difficulty with bodily movements, \
involuntary movements, muscle rigidity, problems with coordination, rhythmic muscle contractions, \
slow bodily movement, or slow shuffling gait. Patient has amnesia, confusion in the evening hours, dementia, or difficulty thinking and understanding"

#control passage
passage5 = "The quick brown fox jumped over the lazy dog. The dog ducked and started barking. The fox ran away"

passages = [passage1, passage2, passage3, passage4, passage5]

#put passages into spacy container for easier preprocessing
nlp = en_core_sci_md.load()
docs = [nlp(passage) for passage in passages]

#declare s-bert embedding model
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

#preprocess strings by separating passage into sentences, removing grammar, encoding each sentence and taking mean to get passage vector
sentence_embeddings = []
for item in docs:
  processed_strings = [str(sent).translate(str.maketrans('', '', string.punctuation)) for sent in list(item.sents)]
  sentence_embeddings.append(np.mean(model.encode([str(sent).translate(str.maketrans('', '', string.punctuation)) for sent in list(item.sents)]), axis = 0))

#calculate cosine similarities for testing of model
lung_base = 1 - cosine(sentence_embeddings[0], sentence_embeddings[4])
lung_lung = 1 - cosine(sentence_embeddings[0], sentence_embeddings[1])
lung_brain = 1 - cosine(sentence_embeddings[0], sentence_embeddings[2])
lung_brain_2 = 1 - cosine(sentence_embeddings[0], sentence_embeddings[3])
brain_brain = 1 - cosine(sentence_embeddings[2], sentence_embeddings[3])


print(lung_lung)
print(lung_brain)
print(lung_brain_2)
print(brain_brain)