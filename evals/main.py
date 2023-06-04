__author__ = 'qiao'

"""
reference:
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=1G6hT73KOzfd
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking import Rerank

from models import DenseRetriever, CrossEncoder # DIY models
import torch

import argparse
import json
import os
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
					datefmt='%Y-%m-%d %H:%M:%S',
					level=logging.INFO,
					handlers=[LoggingHandler()])

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
			"--dataset",
			type=str,
			default="scifact",
			help="The evaluation dataset."
	)
	parser.add_argument(
			"--query_enc_path", 
			type=str, 
			default="malteos/PubMedNCL",
			help="Path to the query encoder."
	)
	parser.add_argument(
			"--doc_enc_path", 
			type=str,
			default="malteos/PubMedNCL",
			help="Path to the document encoder."
	)
	parser.add_argument(
			"--retriever_tokenizer_path", 
			type=str,
			default="malteos/PubMedNCL",
			help="Path to the retriever tokenizer."
	)
	parser.add_argument(
			"--reranking", action='store_true', 
			help="Whether doing re-ranking."
	)
	parser.add_argument(
			"--cross_enc_path",
			type=str,
			default="malteos/PubMedNCL",
			help="Path to the cross encoder."
	)
	parser.add_argument(
			"--reranker_tokenizer_path",
			type=str,
			default="malteos/PubMedNCL",
			help="Path to the cross encoder tokenizer."
	)
	parser.add_argument(
			"--top_k",
			type=int,
			default="100",
			help="The number of top documents to re-rank."
	)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	bi_encoder = DRES(DenseRetriever(args.query_enc_path, args.doc_enc_path, args.retriever_tokenizer_path, device), batch_size=16)
	retriever = EvaluateRetrieval(bi_encoder, score_function="dot")
	
	url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/data/{}.zip".format(args.dataset)
	out_dir = os.path.join(os.getcwd(), "data")
	data_path = util.download_and_unzip(url, out_dir)
	print("Dataset downloaded here: {}".format(data_path))

	data_path = f'data/{args.dataset}'
	corpus, queries, qrels = GenericDataLoader(data_path).load(split="test") # or split = "train" or "dev
	results = retriever.retrieve(corpus, queries)
	output = {'retrieval': EvaluateRetrieval.evaluate(qrels, results, retriever.k_values)}

	if args.reranking:
		cross_encoder = CrossEncoder(args.cross_enc_path, args.reranker_tokenizer_path, device)
		reranker = Rerank(cross_encoder, batch_size=16)
		rerank_results = reranker.rerank(corpus, queries, results, top_k=args.top_k)
		output['reranking'] = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

	with open(f'results/{args.dataset}_results.json', 'w') as f:
		json.dump(output, f, indent=4)
