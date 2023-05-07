__author__ = 'qiao'

'''
Pytorch class for the BioCPT re-ranker
'''

import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification

class CrossEncoder(nn.Module):
	def __init__(self, args):
		super(CrossEncoder, self).__init__()
		self.args = args
		path = args.bert_path
		self.bert = BertForSequenceClassification.from_pretrained(path, num_labels=1)

	
	def save_pretrained(self, path):
		self.bert.save_pretrained(path)
	

	def forward(self, inputs):
		logits = self.bert(input_ids=inputs['input_ids'],
						   attention_mask=inputs['input_mask'],
						   token_type_ids=inputs['segment_ids']).logits # B x 1

		all_probs = F.log_softmax(logits.permute(1, 0), dim=1) # 1 x B
		pos_inds = torch.tensor([0], dtype=torch.long).to(self.args.device)

		loss = F.nll_loss(all_probs,
				pos_inds)

		return loss
