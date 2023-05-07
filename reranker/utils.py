__author__ = 'qiao'

# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Load PubLogBERT dataset. 
'''


import json
import logging
import math
import random
random.seed(2023)

from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class QueryDataset(object):
	
	def __init__(
		self, 
		all_qids,
		all_input_ids,
		all_input_mask,
		all_segment_ids
	):
		self.tensors = [all_input_ids, all_input_mask, all_segment_ids]
		self.qid2idx = {qid:idx for idx, qid in enumerate(all_qids)}

	def __getitem__(self, qid):
		# input_ids, input_mask, segment_ids
		return tuple(torch.tensor(tensor[self.qid2idx[qid]], dtype=torch.long) for tensor in self.tensors) 


class QueryFeatures(object):
	'''
	basically a dict	
	'''
	def __init__(
		self,
		qid,
		input_ids,
		input_mask,
		segment_ids,
	):
		self.qid = qid
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids


def convert_query_to_features(
	qid2info,	
	tokenizer,
	max_query_length
):
	'''Loads a data file into a list of QueryFeatures.'''

	features = []

	#for idx, qid in tqdm(enumerate(qid2info.keys())):
	#for query, qid in tqdm(query2qid.items()):
	for qid, query in tqdm(qid2info.items()):

		tokens, input_ids, input_mask, segment_ids = process_query(
			query, 
			tokenizer, 
			max_query_length
		)
		
		if int(qid) < 20:
			logger.info("\n*** Example ***")
			logger.info("qid: %d" % int(qid))
			logger.info("query tokens: %s" % " ".join(tokens))
			logger.info("query input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("query input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info("query segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

		features.append(
			QueryFeatures(
				qid=qid,
				input_ids=input_ids,
				input_mask=input_mask,
				segment_ids=segment_ids,
			)
		)

	return features


class PubMedDataset(object):
	
	def __init__(
		self, 
		all_pmids,
		all_input_ids,
		all_input_mask,
		all_segment_ids
	):
		self.tensors = [all_input_ids, all_input_mask, all_segment_ids]
		self.pmid2idx = {int(pmid): idx for idx, pmid in enumerate(all_pmids)}

	def __getitem__(self, pmid):
		# input_ids, input_mask, segment_ids
		return tuple(torch.tensor(tensor[self.pmid2idx[pmid]], dtype=torch.long) for tensor in self.tensors) 


class PubMedFeatures(object):
	'''
	basically a dict	
	'''
	def __init__(
		self,
		pmid,
		input_ids,
		input_mask,
		segment_ids,
	):
		self.pmid = pmid
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids


def convert_pubmed_to_features(
	pmid2info,
	tokenizer,
	max_doc_length
):
	'''Loads a data file into a list of PubMedFeatures.'''

	features = []

	for idx, pmid in tqdm(enumerate(pmid2info.keys())):
		
		tokens, input_ids, input_mask, segment_ids = process_doc(
			pmid,
			tokenizer, 
			max_doc_length, 
			pmid2info
		)

		if idx < 20:
			logger.info("\n*** Example ***")
			logger.info("pmid: %d" % int(pmid))
			logger.info("doc tokens: %s" % " ".join(tokens))
			logger.info("doc input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("doc input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info("doc segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

		features.append(
			PubMedFeatures(
				pmid=pmid,
				input_ids=input_ids,
				input_mask=input_mask,
				segment_ids=segment_ids,
			)
		)

	return features



def read_train_examples(input_file): 
	'''
	read the training dataset and convert them to examples
	'''

	with open(input_file, "r", encoding="utf-8") as f:
		lines = f.readlines()

	examples = []

	for line in lines:
		entry = json.loads(line)
		
		example = TrainExample(
				qid=int(entry['qid']),
				pmid=int(entry['pmid']),
				click=entry['click']
		)

		examples.append(example)

	return examples



def process_query(query, tokenizer, max_len):
	ori_tokens = tokenizer.tokenize(query)

	tokens = []
	tokens += ['[CLS]']
	tokens += ori_tokens[:(max_len - 2)]
	tokens += ['[SEP]']

	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	segment_ids = [0] * len(tokens) 
	input_mask = [1] * len(tokens)

	# Zero-pad up to the sequence length
	while len(input_ids) < max_len:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	assert len(input_ids) == max_len
	assert len(input_mask) == max_len
	assert len(segment_ids) == max_len

	return tokens, input_ids, input_mask, segment_ids


def process_doc(pmid, tokenizer, max_len, pmid2info):
	if len(pmid2info[pmid]) < 2:
		pmid2info[pmid] += [''] * (2 - len(pmid2info[pmid]))

	ti_tokens = tokenizer.tokenize(pmid2info[pmid][0])	
	ab_tokens = tokenizer.tokenize(pmid2info[pmid][1])

	tiab_tokens = ti_tokens + ab_tokens

	tokens = []
	tokens += tiab_tokens[:(max_len - 1)]
	tokens += ['[SEP]']

	segment_ids = [1] * len(tokens) 
	input_mask = [1] * len(tokens)
	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# Zero-pad up to the sequence length
	while len(input_ids) < max_len:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(1)

	assert len(input_ids) == max_len
	assert len(input_mask) == max_len
	assert len(segment_ids) == max_len

	return tokens, input_ids, input_mask, segment_ids
