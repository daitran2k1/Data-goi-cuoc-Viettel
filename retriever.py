"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


#Define our Cross-Encoder
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_stsbenchmark-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#We use phobert-base as base model and set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder('vinai/phobert-base', num_labels=1)


# Read STSb dataset
logger.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []

with open('prompt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
  
trainval_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(trainval_data, test_size=0.2, random_state=42)
for sample in train_data:
    train_samples.append(InputExample(texts=[sample[0], sample[1]], label=1))
    train_samples.append(InputExample(texts=[sample[1], sample[0]], label=1))
for sample in val_data:
    dev_samples.append(InputExample(texts=[sample[0], sample[1]], label=1))
for sample in test_data:
    test_samples.append(InputExample(texts=[sample[0], sample[1]], label=1))


# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##### Load model and eval on test set
model = CrossEncoder(model_save_path)

evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='sts-test')
evaluator(model)

# from sentence_transformers.cross_encoder import CrossEncoder
# import numpy as np

# # Pre-trained cross encoder
# model = CrossEncoder('cross-encoder/stsb-distilroberta-base')

# # We want to compute the similarity between the query sentence
# query = 'A man is eating pasta.'

# # With all sentences in the corpus
# corpus = ['A man is eating food.',
#           'A man is eating a piece of bread.',
#           'The girl is carrying a baby.',
#           'A man is riding a horse.',
#           'A woman is playing violin.',
#           'Two men pushed carts through the woods.',
#           'A man is riding a white horse on an enclosed ground.',
#           'A monkey is playing drums.',
#           'A cheetah is running behind its prey.'
#           ]

# # So we create the respective sentence combinations
# sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

# # Compute the similarity scores for these combinations
# similarity_scores = model.predict(sentence_combinations)

# # Sort the scores in decreasing order
# sim_scores_argsort = reversed(np.argsort(similarity_scores))

# # Print the scores
# print("Query:", query)
# for idx in sim_scores_argsort:
#     print("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))