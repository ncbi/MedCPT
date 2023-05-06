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
#
# modified from https://github.com/huggingface/transformers/blob/6e1ee47b361f9dc1b6da0104d77b38297042efae/examples/legacy/question-answering/run_squad.py
# author: Qiao Jin

import argparse
import json
import logging
import os
import random

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange

import utils
import models
import numpy as np

import transformers
from transformers import (
	AdamW,
	BertConfig,
	BertTokenizer,
	get_linear_schedule_with_warmup,
)
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, qid2info, pmid2info, biencoder, tokenizer):
	""" Train the model """
	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

	t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in biencoder.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in biencoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	# multi-gpu training 
	if args.n_gpu > 1:
		biencoder = torch.nn.DataParallel(biencoder)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. accumulation) = %d",
		args.train_batch_size
		* args.gradient_accumulation_steps)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	biencoder.zero_grad()

	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
	set_seed(args)	# Added here for reproductibility

	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)

		for step, batch in enumerate(epoch_iterator):
			biencoder.train()

			qids = batch['qid'] 
			pmids = batch['pmid']

			query_batch = [qid2info[qid] for qid in qids] 
			paper_batch = [pmid2info[pmid] for pmid in pmids]

			queries = tokenizer(query_batch, truncation=True, padding=True, return_tensors='pt', max_length=args.max_query_length)
			queries.to(args.device)
			q_input_ids = queries['input_ids']
			q_token_type_ids = queries['token_type_ids']
			q_attention_mask = queries['attention_mask']

			papers = tokenizer(paper_batch, truncation=True, padding=True, return_tensors='pt', max_length=args.max_doc_length)
			papers.to(args.device)
			d_input_ids = papers['input_ids']
			d_token_type_ids = papers['token_type_ids']
			d_attention_mask = papers['attention_mask']

			weights = batch['click'].float().to(args.device)

			loss = biencoder(q_input_ids, q_token_type_ids, q_attention_mask,
					d_input_ids, d_token_type_ids, d_attention_mask, weights)

			if args.n_gpu > 1:
				loss = loss.mean()	# mean() to average on multi-gpu parallel (not distributed) training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			loss.backward()
			tr_loss += loss.item()

			if (step + 1) % args.gradient_accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(biencoder.parameters(), args.max_grad_norm)

				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				biencoder.zero_grad()
				global_step += 1

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
						biencoder.module if hasattr(biencoder, "module") else biencoder 
					)  # Take care of distributed/parallel training
					model_to_save.save_pretrained(output_dir)
					
					logger.info("Saving model checkpoint to %s", output_dir)


	return global_step, tr_loss / global_step


def main():
	parser = argparse.ArgumentParser()

	# Path parameters
	parser.add_argument(
		'--bert_q_path',
		default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
		type=str,
		help='The path of the pre-trained query encoder.'
	)
	parser.add_argument(
		'--bert_d_path',
		default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
		type=str,
		help='The path of the pre-trained document encoder.'
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
	# Dataset paths
	parser.add_argument(
		'--train_dataset',
		default=None,
		type=str,
		help='The path of the training jsonline file.'
	)
	parser.add_argument(
		'--pmid2info_path', default=None, type=str, help="The pmid2info json file path."
	)
	parser.add_argument(
		'--qid2info_path', default=None, type=str, help="The qid2info json file path."
	)

	# Hyper-parameters
	parser.add_argument(
		"--max_query_length",
		default=64,
		type=int,
		help="Maximum length of query."
	)
	parser.add_argument(
		"--max_doc_length",
		default=512,
		type=int,
		help="Maximum length of documents."
	)
	parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--alpha", default=0.8, type=float, help="Alpha in the loss combination.")
	parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
		"--num_train_epochs", default=8.0, type=float, help="Total number of training epochs to perform."
	)
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument("--warmup_steps", default=10000, type=int, help="Linear warmup over warmup_steps.")

	# Logging and saving steps
	parser.add_argument("--logging_steps", type=int, default=25, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=2500, help="Save checkpoint every X updates steps.")
	parser.add_argument("--do_lower_case", default=True, type=int, help="Set this flag if you are using an uncased model. Queries are uncased, so setting default to True.")
	parser.add_argument("--seed", type=int, default=2023, help="random seed for initialization")

	# parse the arguments
	args = parser.parse_args()

	# Create output directory if needed
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Setup CUDA, GPU 
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

	# save the args before actual training
	torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

	logger.info("Loading dataset, qid2info, and pmid2info.")
	train_dataset = utils.PairDataset(args.train_dataset) 
	qid2info = json.load(open(args.qid2info_path))
	pmid2info = json.load(open(args.pmid2info_path))	

	biencoder = models.Biencoder(args)
	biencoder.to(args.device)

	# actual training
	global_step, tr_loss = train(args, train_dataset, qid2info, pmid2info, biencoder, tokenizer) 
	logger.info("Global_step = %s, average loss = %s", global_step, tr_loss)

	# saving the model and tokenizer
	logger.info("Saving model checkpoint to %s", args.output_dir)
	biencoder.save_pretrained(args.output_dir)
	tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
	main()
