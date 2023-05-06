__author__ = 'qiao'

'''
BioCPT utilities
'''

import json
import logging
import math
import torch
logger = logging.getLogger(__name__)

class PairDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_path):
		'''
		dataset_path: path to a training jsonline file
		'''
		self.examples = []

		with open(dataset_path, "r", encoding="utf-8") as f:
			lines = f.readlines()

		for line in lines:
			entry = json.loads(line)
			entry['qid'] = str(entry['qid'])
			entry['pmid'] = str(entry['pmid'])
			entry['click'] = math.log(entry['click'] + 1, 2)
			self.examples.append(entry)
	
	def __getitem__(self, index):
		return self.examples[index]
	
	def __len__(self):
		return len(self.examples)
