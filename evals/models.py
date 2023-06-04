__author__ = 'qiao'

'''
modified from https://github.com/beir-cellar/beir/wiki/Evaluate-your-custom-model
'''

from transformers import AutoModel, BertForSequenceClassification, AutoTokenizer
import torch
from tqdm import trange
import os

class DenseRetriever:
	def __init__(self, query_enc_path, doc_enc_path, tokenizer_path, device):
		self.device = device

		self.bert_q = AutoModel.from_pretrained(query_enc_path)
		self.bert_q.eval()
		self.bert_q.to(self.device)

		self.bert_d = AutoModel.from_pretrained(doc_enc_path)
		self.bert_d.eval()
		self.bert_d.to(self.device)
		
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) 

	def encode_queries(self, queries, batch_size=16, **kwargs):
		query_embeddings = []

		with torch.no_grad():
			for start_idx in trange(0, len(queries), batch_size):
				encoded = self.tokenizer(queries[start_idx:start_idx+batch_size], truncation=True, padding=True, return_tensors='pt', max_length=512)
				encoded.to(self.device)
				model_out = self.bert_q(**encoded)
				query_embeddings += model_out.last_hidden_state[:, 0, :].detach().cpu()

		return torch.stack(query_embeddings)
		
	def encode_corpus(self, corpus, batch_size=16, **kwargs):
		corpus_embeddings = []

		with torch.no_grad():
			for start_idx in trange(0, len(corpus), batch_size):
				titles = [row['title'] for row in corpus[start_idx: start_idx + batch_size]]
				texts = [row['text']  for row in corpus[start_idx: start_idx + batch_size]]
				encoded = self.tokenizer(titles, texts, truncation='longest_first', padding=True, return_tensors='pt', max_length=512)
				encoded.to(self.device)
				model_out = self.bert_d(**encoded)
				corpus_embeddings += model_out.last_hidden_state[:, 0, :].detach().cpu()
		
		return torch.stack(corpus_embeddings)


class CrossEncoder:
	def __init__(self, cross_enc_path, tokenizer_path, device):
		self.device = device

		self.cross_enc = BertForSequenceClassification.from_pretrained(cross_enc_path, num_labels=1)
		self.cross_enc.eval()
		self.cross_enc.to(self.device)

		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) 
	
	def predict(self, sentences, batch_size, **kwargs):
		scores = []

		with torch.no_grad():
			for start_idx in trange(0, len(sentences), batch_size):
				batch_sentences = sentences[start_idx: start_idx + batch_size]
				encoded = self.tokenizer(batch_sentences, truncation='longest_first', padding=True, return_tensors='pt', max_length=512)
				encoded.to(self.device)
				model_out = self.cross_enc(**encoded).logits.squeeze(dim=1)
				scores += list(model_out.detach().cpu().numpy())

		return scores
