import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import f1_score, accuracy_score
from wordcloud import WordCloud, STOPWORDS
from collections import Counter, defaultdict
from PIL import Image
import spacy
import en_core_web_sm

# Core packages for general use throughout the notebook.

import random
import warnings
import time
import datetime
#------------------------------
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
#------------------------
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler



# Setting some options for general use.

stop = set(stopwords.words('english'))
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.5)
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
warnings.filterwarnings('ignore')


#Setting seeds for consistent results.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def predict(combined = ['FLYNN: Hillary Clinton, Big Woman on Campus - Breitbart Daniel J. Flynn']):
    if torch.cuda.is_available():
        device = torch.device('cuda')    
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')



    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

    max_len = 0
    for text in combined:

        input_ids = tokenizer.encode(text, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)

    token_lens = []

    for text in combined:
        tokens = tokenizer.encode(text, max_length = 512)
        token_lens.append(len(tokens))


    test = combined

    def tokenize_map(sentence,labs='None'):
        
        """A function for tokenize all of the sentences and map the tokens to their word IDs."""
        
        global labels
        
        input_ids = []
        attention_masks = []
        
        for text in sentence:
            encoded_dict = tokenizer.encode_plus(
                                text,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                truncation='longest_first', # Activate and control truncation
                                max_length = 84,           # Max length according to our text data.
                                pad_to_max_length = True, # Pad & truncate all sentences.
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        if labs != 'None': # Setting this for using this definition for both train and test data so labels won't be a problem in our outputs.
            labels = torch.tensor(labels)
            return input_ids, attention_masks, labels
        else:
            return input_ids, attention_masks



    test_input_ids, test_attention_masks= tokenize_map(test)

    batch_size = 15

    prediction_data = TensorDataset(test_input_ids, test_attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


    model = BertForSequenceClassification.from_pretrained(
        'bert-large-uncased', # Use the 124-layer, 1024-hidden, 16-heads, 340M parameters BERT model with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification. You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model.to(device)


    model.load_state_dict(torch.load("fake-news-detector.pth"))


    print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))
    model.eval()
    predictions = []

    for batch in prediction_dataloader: 
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)


    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    return flat_predictions[0]

# print(predict())
