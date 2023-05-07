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
# Author: Qiao Jin

import argparse
import glob
import json
import logging
import os
import random

import math
import numpy as np
import torch
from tqdm import tqdm, trange

from utils import (
	PubMedDataset,
	convert_pubmed_to_features,
	QueryDataset,
	convert_query_to_features
)

from transformers import (
	AdamW,
	BertTokenizer,
	get_linear_schedule_with_warmup,
)

import models

logger = logging.getLogger(__name__)

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, query_dataset, pubmed_dataset, model):
	""" Train the model """
	t_total = len(train_dataset) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	# multi-gpu training 
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Train
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info(
		"  Total train batch size = Accumulation = %d",
		args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	sum_weight = 0

	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
	set_seed(args)	# Added here for reproductibility

	for _ in train_iterator:
		random.shuffle(train_dataset)

		pos_pmids = []
		weights = []
		
		for train_entry in train_dataset:
			pos_pmid, weight = random.choice(train_entry['pos_pmids'])
			pos_pmids.append(pos_pmid)
			weights.append(math.log(weight + 1, 2))
		
		model.zero_grad()
		sum_weight = sum(weights[:args.gradient_accumulation_steps])

		for step, (pos_pmid, weight, train_entry) in tqdm(enumerate(zip(pos_pmids, weights, train_dataset))):
			if len(train_entry['neg_pmids']) <= args.num_neg_pmids: continue
			model.train()
			
			qid = train_entry['qid']
			pos_pmid = [int(pos_pmid)]
			neg_pmids = random.sample(train_entry['neg_pmids'], k=args.num_neg_pmids)
			neg_pmids = [int(neg_pmid) for neg_pmid in neg_pmids]

			query_batch = [query_dataset[qid] for _ in range(1 + args.num_neg_pmids)] 
			query_batch = list(map(list, zip(*query_batch)))

			paper_batch = [pubmed_dataset[pmid] for pmid in pos_pmid + neg_pmids]
			paper_batch = list(map(list, zip(*paper_batch)))

			query_batch = tuple(torch.stack(t).to(args.device) for t in query_batch)
			paper_batch = tuple(torch.stack(t).to(args.device) for t in paper_batch)

			inputs = {
				'mode': 'training',
				'input_ids': torch.cat([query_batch[0], paper_batch[0]], dim=1),
				'input_mask': torch.cat([query_batch[1], paper_batch[1]], dim=1),
				'segment_ids': torch.cat([query_batch[2], paper_batch[2]], dim=1)
			}

			loss = model(inputs)
			loss = loss * weight / sum_weight

			loss.backward()
			tr_loss += loss.item()

			if (step + 1) % args.gradient_accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				sum_weight = sum(weights[step + 1: step + 1 + args.gradient_accumulation_steps])  

				if args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					logging_loss = tr_loss / global_step
					logger.info("Logging_loss: %.4f" % logging_loss)

				if args.save_steps > 0 and global_step % args.save_steps == 0:
					# Save model checkpoint
					output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					model_to_save = (
						model.module if hasattr(model, "module") else model 
					)  # Take care of distributed/parallel training
					model_to_save.save_pretrained(output_dir)
					
					logger.info("Saving model checkpoint to %s", output_dir)


	return global_step, tr_loss / global_step


def load_and_cache_pubmed(args, tokenizer):
	input_file = args.pmid2info_path
	basename = os.path.basename(input_file)

	cached_features_file = os.path.join(
		os.path.dirname(input_file),
		'cached_{}_dlen{}'.format(
			basename,
			args.max_doc_length
		),
	)

	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
		logger.info("Creating features from dataset file at %s", input_file)
		logger.info("Loading the pmid2info")

		pmid2info = json.load(open(input_file))

		features = convert_pubmed_to_features(
			pmid2info=pmid2info,
			tokenizer=tokenizer,
			max_doc_length=args.max_doc_length
		)

		logger.info("Saving features into cached file %s", cached_features_file)
		torch.save(features, cached_features_file)
	
	# Convert to Tensors and build dataset
	all_pmids = [f.pmid for f in features]
	all_input_ids = [f.input_ids for f in features]
	all_input_mask = [f.input_mask for f in features]
	all_segment_ids = [f.segment_ids for f in features]

	dataset = PubMedDataset(
		all_pmids,
		all_input_ids,
		all_input_mask,
		all_segment_ids
	)

	return dataset


def load_and_cache_query(args, tokenizer):
	input_file = args.qid2info_path
	#input_file = args.query2qid_path
	basename = os.path.basename(input_file)

	cached_features_file = os.path.join(
		os.path.dirname(input_file),
		'cached_{}_dlen{}'.format(
			basename,
			args.max_query_length
		),
	)

	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
		logger.info("Creating features from dataset file at %s", input_file)
		logger.info("Loading the qid2info")
		qid2info  = json.load(open(input_file))

		features = convert_query_to_features(
			qid2info=qid2info,
			tokenizer=tokenizer,
			max_query_length=args.max_query_length
		)

		logger.info("Saving features into cached file %s", cached_features_file)
		torch.save(features, cached_features_file)
	
	# Convert to Tensors and build dataset
	all_qids = [f.qid for f in features]
	all_input_ids = [f.input_ids for f in features]
	all_input_mask = [f.input_mask for f in features]
	all_segment_ids = [f.segment_ids for f in features]

	dataset = QueryDataset(
		all_qids,
		all_input_ids,
		all_input_mask,
		all_segment_ids
	)

	return dataset


def main():
	parser = argparse.ArgumentParser()


	# Path parameters
	parser.add_argument(
		'--bert_path',
		default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
		type=str,
		help='The path of the pre-trained query encoder.'
	)
	parser.add_argument(
		'--tokenizer_path',
		default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
		help='The path of the tokenizer.'
	)

	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		required=True,
		help="The output directory where the model checkpoints and predictions will be written.",
	)
	parser.add_argument(
		'--pmid2info_path', default=None, type=str, help="The path to pmid2info json file."
	)
	parser.add_argument(
		'--qid2info_path', default=None, type=str, help="The path to qid2info json file."
	)
	parser.add_argument(
		'--train_dataset',
		default=None,
		type=str,
		help='The path of the training dataset.'
	)
	
	# Hyperparameters
	parser.add_argument(
		"--max_query_length",
		default=32,
		type=int,
		help="max length of query."
	)
	parser.add_argument(
		"--max_doc_length",
		default=480,
		type=int,
		help="max length of documents."
	)

	parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--num_neg_pmids", default=31, type=int, help="Negative pmids per batch")
	parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.")
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=32,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
		"--num_train_epochs", default=8.0, type=float, help="Total number of training epochs to perform."
	)
	parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument("--warmup_steps", default=10000, type=int, help="Linear warmup over warmup_steps.")

	# Logging and saving steps
	parser.add_argument("--logging_steps", type=int, default=25, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=2500, help="Save checkpoint every X updates steps.")

	# others
	parser.add_argument(
		"--eval_all_checkpoints",
		action="store_true",
		help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
	)
	parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
	)

	# you might not want to change these
	parser.add_argument("--do_lower_case", default=True, type=int, help="Set this flag if you are using an uncased model. Queries are uncased, so setting default to True..")
	parser.add_argument("--seed", type=int, default=2023, help="random seed for initialization")

	# parse the arguments
	args = parser.parse_args()

	# Create output directory if needed
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Setup CUDA, GPU 
	device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
	args.n_gpu = torch.cuda.device_count()
	args.device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO
	)
	logger.warning(
		"Process device: %s, n_gpu: %s",
		device,
		args.n_gpu
	)

	# Set seed
	set_seed(args)

	# Set tokenizer
	tokenizer = BertTokenizer.from_pretrained(
		args.tokenizer_path,
		do_lower_case=args.do_lower_case
	)

	logger.info("Script parameters %s", args)

	# Training
	# save the args before actual training
	torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
	
	train_dataset = json.load(open(args.train_dataset))
	query_dataset = load_and_cache_query(args, tokenizer)
	pubmed_dataset = load_and_cache_pubmed(args, tokenizer)

	model = models.CrossEncoder(args)
	model.to(args.device)

	# actual training
	global_step, tr_loss = train(args, train_dataset, query_dataset, pubmed_dataset, model) 
	logger.info("Global_step = %s, average loss = %s", global_step, tr_loss)

	# saving the model and tokenizer
	logger.info("Saving model checkpoint to %s", args.output_dir)
	model.save_pretrained(args.output_dir)
	tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
	main()
