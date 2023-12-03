# MedCPT: Zero-shot Biomedical IR Model

## Overview

![image](https://github.com/ncbi/MedCPT/assets/32558774/6c1bde8d-1930-4df0-a120-16bb7f0e0d3a)

MedCPT is a first-of-its-kind Contrastive Pre-trained Transformer model trained with an unprecedented scale of PubMed search logs for zero-shot biomedical information retrieval. MedCPT consists of:
- A frist-stage dense retriever (MedCPT retriever)
  - Contains a query encoder (QEnc) and an article encoder (DEnc), both initialized by [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).   
  - Trained by 255M query-article pairs from PubMed search logs and in-batch negatives. 
- A second-stage re-ranker (MedCPT re-ranker)
  - A transformer cross-encoder (CrossEnc) initialized by [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).
  - Trained by 18M semantic query-article pairs and localized negatives from the pre-trained MedCPT retriever. 

## Training MedCPT

This directory contains:
- [Code](./retriever/) for training the MedCPT retriever.
- [Code](./reranker/) for training the MedCPT re-ranker.
- [Code](./evals/) for evaluating the pre-trained model.

## Using MedCPT

MedCPT model weights are publicly available on Hugging Face:
- [MedCPT Query Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder)
- [MedCPT Article Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder)
- [MedCPT Cross Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder)

### Using the MedCPT Query Encoder

```python
import torch
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

queries = [
	"diabetes treatment", 
	"How to treat diabetes?", 
	"A 45-year-old man presents with increased thirst and frequent urination over the past 3 months.",
]

with torch.no_grad():
	# tokenize the queries
	encoded = tokenizer(
		queries, 
		truncation=True, 
		padding=True, 
		return_tensors='pt', 
		max_length=64,
	)
	
	# encode the queries (use the [CLS] last hidden states as the representations)
	embeds = model(**encoded).last_hidden_state[:, 0, :]

	print(embeds)
	print(embeds.size())
```
The output will be:
```bash
tensor([[ 0.0413,  0.0084, -0.0491,  ..., -0.4963, -0.3830, -0.3593],
        [ 0.0801,  0.1193, -0.0905,  ..., -0.5380, -0.5059, -0.2944],
        [-0.3412,  0.1521, -0.0946,  ...,  0.0952,  0.1660, -0.0902]])
torch.Size([3, 768])
```
These embeddings are also in the same space as those generated by the MedCPT article encoder. 

### Using the MedCPT Article Encoder
```python
import torch
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

# each article contains a list of two texts (usually a title and an abstract)
articles = [
	[
		"Diagnosis and Management of Central Diabetes Insipidus in Adults",
		"Central diabetes insipidus (CDI) is a clinical syndrome which results from loss or impaired function of vasopressinergic neurons in the hypothalamus/posterior pituitary, resulting in impaired synthesis and/or secretion of arginine vasopressin (AVP). [...]",
	],
	[
		"Adipsic diabetes insipidus",
		"Adipsic diabetes insipidus (ADI) is a rare but devastating disorder of water balance with significant associated morbidity and mortality. Most patients develop the disease as a result of hypothalamic destruction from a variety of underlying etiologies. [...]",
	],
	[
		"Nephrogenic diabetes insipidus: a comprehensive overview",
		"Nephrogenic diabetes insipidus (NDI) is characterized by the inability to concentrate urine that results in polyuria and polydipsia, despite having normal or elevated plasma concentrations of arginine vasopressin (AVP). [...]",
	],
]

with torch.no_grad():
	# tokenize the queries
	encoded = tokenizer(
		articles, 
		truncation=True, 
		padding=True, 
		return_tensors='pt', 
		max_length=512,
	)
	
	# encode the queries (use the [CLS] last hidden states as the representations)
	embeds = model(**encoded).last_hidden_state[:, 0, :]

	print(embeds)
	print(embeds.size())
```
The output will be:
```bash
tensor([[-0.0189,  0.0115,  0.0988,  ..., -0.0655,  0.3155, -0.0357],
        [-0.3402, -0.3064, -0.0749,  ..., -0.0799,  0.3332,  0.1263],
        [-0.2764, -0.0506, -0.0608,  ...,  0.0389,  0.2532,  0.1580]])
torch.Size([3, 768])
```
These embeddings are also in the same space as those generated by the MedCPT query encoder. 

### Using the MedCPT Cross Encoder
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")

query = "diabetes treatment"

# 6 articles to be ranked for the input query
articles = [
	"Type 1 and 2 diabetes mellitus: A review on current treatment approach and gene therapy as potential intervention. Type 1 and type 2 diabetes mellitus is a serious and lifelong condition commonly characterised by abnormally elevated blood glucose levels due to a failure in insulin production or a decrease in insulin sensitivity and function. [...]",
	"Diabetes mellitus and its chronic complications. Diabetes mellitus is a major cause of morbidity and mortality, and it is a major risk factor for early onset of coronary heart disease. Complications of diabetes are retinopathy, nephropathy, and peripheral neuropathy. [...]",
	"Diagnosis and Management of Central Diabetes Insipidus in Adults. Central diabetes insipidus (CDI) is a clinical syndrome which results from loss or impaired function of vasopressinergic neurons in the hypothalamus/posterior pituitary, resulting in impaired synthesis and/or secretion of arginine vasopressin (AVP). [...]",
	"Adipsic diabetes insipidus. Adipsic diabetes insipidus (ADI) is a rare but devastating disorder of water balance with significant associated morbidity and mortality. Most patients develop the disease as a result of hypothalamic destruction from a variety of underlying etiologies. [...]",
	"Nephrogenic diabetes insipidus: a comprehensive overview. Nephrogenic diabetes insipidus (NDI) is characterized by the inability to concentrate urine that results in polyuria and polydipsia, despite having normal or elevated plasma concentrations of arginine vasopressin (AVP). [...]",
	"Impact of Salt Intake on the Pathogenesis and Treatment of Hypertension. Excessive dietary salt (sodium chloride) intake is associated with an increased risk for hypertension, which in turn is especially a major risk factor for stroke and other cardiovascular pathologies, but also kidney diseases. Besides, high salt intake or preference for salty food is discussed to be positive associated with stomach cancer, and according to recent studies probably also obesity risk. [...]"
]

# combine query article into pairs
pairs = [[query, article] for article in articles]

with torch.no_grad():
	encoded = tokenizer(
		pairs,
		truncation=True,
		padding=True,
		return_tensors="pt",
		max_length=512,
	)

	logits = model(**encoded).logits.squeeze(dim=1)
	
	print(logits)
```

The output will be
```bash
tensor([  6.9363,  -8.2063,  -8.7692, -12.3450, -10.4416, -15.8475])
```
Higher scores indicate higher relevance.

## Data availability

Due to [privacy](https://www.nlm.nih.gov/web_policies.html#privacy_security) concerns, we are not able to release the PubMed user logs. As a surrogate, we provide the question-article pair data from [BioASQ](http://www.bioasq.org/) in this repo as example training datasets. You can convert your data to the example data formats and train the MedCPT model.

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.

## Citation

If you find this repo helpful, please cite MedCPT by:

```bibtext
@article{jin2023medcpt,
  title={MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval},
  author={Jin, Qiao and Kim, Won and Chen, Qingyu and Comeau, Donald C and Yeganova, Lana and Wilbur, W John and Lu, Zhiyong},
  journal={Bioinformatics},
  volume={39},
  number={11},
  pages={btad651},
  year={2023},
  publisher={Oxford University Press}
}
```
