__author__ = 'qiao'

'''
The BioCPT model class
'''

import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class Biencoder(nn.Module): 
	def __init__(self, args):
		super(Biencoder, self).__init__()
		self.args = args

		q_path = args.bert_q_path
		d_path = args.bert_d_path

		self.config_q = BertConfig.from_pretrained(q_path)
		self.bert_q = BertModel.from_pretrained(q_path)

		self.config_d = BertConfig.from_pretrained(d_path)
		self.bert_d = BertModel.from_pretrained(d_path)


	def save_pretrained(self, path):
		self.config_q.save_pretrained(os.path.join(path, 'query_encoder'))
		self.bert_q.save_pretrained(os.path.join(path, 'query_encoder'))

		self.config_d.save_pretrained(os.path.join(path, 'doc_encoder'))
		self.bert_d.save_pretrained(os.path.join(path, 'doc_encoder'))


	def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
					d_input_ids, d_token_type_ids, d_attention_mask, weights):
		embed_q = self.bert_q(input_ids=q_input_ids,
							  attention_mask=q_attention_mask,
							  token_type_ids=q_token_type_ids)[0][:, 0, :] # B x D

		embed_d = self.bert_d(input_ids=d_input_ids,
							  attention_mask=d_attention_mask,
							  token_type_ids=d_token_type_ids)[0][:, 0, :] # B x D

		B = embed_q.size(dim=0)
		qd_scores = torch.matmul(embed_q, torch.transpose(embed_d, 1, 0)) # B x B

		# q to d softmax
		q2d_softmax = F.log_softmax(qd_scores, dim=1)

		# d to q softmax
		d2q_softmax = F.log_softmax(qd_scores, dim=0)
		
		# positive indices (diagonal)
		pos_inds = torch.tensor(list(range(B)), dtype=torch.long).to(self.args.device)
		
		q2d_loss = F.nll_loss(q2d_softmax,
			pos_inds,
			weight=weights,
			reduction="mean"
		)

		d2q_loss = F.nll_loss(d2q_softmax,
			pos_inds,
			weight=weights,
			reduction="mean"
		)

		loss = self.args.alpha * q2d_loss + (1 - self.args.alpha) * d2q_loss

		return loss
