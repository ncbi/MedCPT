# Evaluations

Evaluate the BioCPT models on biomedical IR datasets with [BEIR](https://github.com/beir-cellar/beir).

## Requirements

The code has been tested with Python 3.9.13. Please first instlal the required packages by:
```bash
pip install -r requirements.txt
```

## Usage

Users can run `main.py` to evaluate a given model (specified by its path) on a dataset (`scifact`, `scidocs`, `trec-covid`, `nfcorpus`). 

```bash
$ python main.py --help
usage: main.py [-h] [--dataset DATASET] [--query_enc_path QUERY_ENC_PATH] [--doc_enc_path DOC_ENC_PATH]
               [--retriever_tokenizer_path RETRIEVER_TOKENIZER_PATH] [--reranking] [--cross_enc_path CROSS_ENC_PATH]
               [--reranker_tokenizer_path RERANKER_TOKENIZER_PATH] [--top_k TOP_K]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     The evaluation dataset.
  --query_enc_path QUERY_ENC_PATH
                        Path to the query encoder.
  --doc_enc_path DOC_ENC_PATH
                        Path to the document encoder.
  --retriever_tokenizer_path RETRIEVER_TOKENIZER_PATH
                        Path to the retriever tokenizer.
  --reranking           Whether doing re-ranking.
  --cross_enc_path CROSS_ENC_PATH
                        Path to the cross encoder.
  --reranker_tokenizer_path RERANKER_TOKENIZER_PATH
                        Path to the cross encoder tokenizer.
  --top_k TOP_K         The number of top documents to re-rank.
```
