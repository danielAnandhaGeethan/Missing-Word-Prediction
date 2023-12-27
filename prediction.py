import torch
import numpy as np
from pytorch_transformers import BertTokenizer, BertForMaskedLM
import nltk
import string
#nltk.download('punkt')

import sys

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
model.eval()

def predict(sentence_orig):
    if '_' not in sentence_orig:
        return sentence_orig

    sentence = sentence_orig.replace('_', '[MASK]')
    tokens = nltk.word_tokenize(sentence)
    sentences = nltk.sent_tokenize(sentence)
    if len(sentences)>2:
        concat = sentences[1:]
        concat = ' '.join([x[:-1] for x in concat])
        sentences = [sentences[0]] + [concat + '.']
    sentence = " [SEP] ".join(sentences)
    sentence = "[CLS] " + sentence + " [SEP]"


    while '[MASK]' in sentence:
      tokenized_text = tokenizer.tokenize(sentence)
      masked_index = tokenized_text.index('[MASK]')
      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

      segments_ids = []

      segment=0
      for token in tokenized_text:
        segments_ids.append(segment)
        if token == '[SEP]':
          segment += 1


      tokens_tensor = torch.tensor([indexed_tokens])
      segments_tensors = torch.tensor([segments_ids])


      with torch.no_grad():
              outputs = model(tokens_tensor, token_type_ids=segments_tensors)
              predictions = outputs[0]
              attention = outputs[-1]

      dim = attention[2][0].shape[-1]*attention[2][0].shape[-1]
      a = attention[2][0].reshape(12, dim)
      b = a.mean(axis=0)
      c = b.reshape(attention[2][0].shape[-1],attention[2][0].shape[-1])
      avg_wgts = c[masked_index]

      predicted_index = torch.argmax(predictions[0, masked_index]).item()
      predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
      sentence = sentence.replace('[MASK]', predicted_token, 1)
      sentence_orig = sentence_orig.replace('_', predicted_token,1)

    return [predicted_token, sentence_orig]

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('_', tokenizer.mask_token)
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]

    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    
    return bert


if __name__ == "__main__":
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

    top_k = 10

    sentence_orig = sys.argv[1:][0]

    if sentence_orig[-1] == ".":
        sentence_orig = sentence_orig[0::-1]

    if "_" not in sentence_orig:
        sentence_orig += " _."

    x = predict(sentence_orig)

    y = get_all_predictions(sentence_orig)
    
    x.append(y)
    print(x)