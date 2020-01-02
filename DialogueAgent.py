import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import flask
from flask import request
from text_emojize import get_emojized_text

app = flask.Flask(__name__)
app.config["DEBUG"] = True


# import huggingface transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values,
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'),
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    return logits




# load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("DialoGPT-medium")

# load the model
model=GPT2LMHeadModel.from_pretrained("DialoGPT-medium")


#use cuda/gpu if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


eos = [tokenizer.encoder["<|endoftext|>"]]

past = None
temperature = 0.9
top_k = -1
top_p = 0.9

model.eval()
prev_input = None


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Dialogue API</h1>
<p>A dialogue API based on a transformer model.</p>'''


@app.route('/api/getresponse', methods=['GET'])
def getresponse():
    global past
    user = request.args.get('query')
    with torch.no_grad():
        user = tokenizer.encode(user)
        prev_input = user
        prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
        _, past = model(prev_input, past=past)

        prev_input = torch.LongTensor([eos]).to(device)

        sent = []
        for i in range(500):
            logits, past = model(prev_input, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_filtering(logits, top_k=top_k, top_p=top_p)

            probs = torch.softmax(logits, dim=-1)

            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == eos[0]:
                break
            sent.append(prev_word)
        prev_input = torch.LongTensor([eos]).to(device)
        _, past = model(prev_input, past=past)
        print(tokenizer.decode(sent))
        if len(sent)>0:
            return get_emojized_text(tokenizer.decode(sent))
        else:
            return '|No Response|'
app.run()
