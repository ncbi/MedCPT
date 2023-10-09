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

## Content

This directory contains:
- MedCPT API [documentation](https://github.com/ncbi/MedCPT/tree/main#MedCPT-api).
- MedCPT weights request [form](https://docs.google.com/forms/d/e/1FAIpQLSdtd2OmkI3ep_RadoiTxdVSqvR7rSDssDlAyrpQDaddhv5oOw/viewform?usp=sf_link).
- Code for training the MedCPT [retriever](./retriever/).
- Code for training the MedCPT [re-ranker](./reranker/).
- Code for evaluating the pre-trained model at [evals](./evals/).

## MedCPT API

We provide 4 endpoints for using the MedCPT model (currently only the retriever):
1. `query2vector`: Returns a dense vector for the given free-text query.
2. `doc2vector`: Returns a dense vector for the given free-text document.
3. `docid2vector`: Returns a dense vector for the given corpus and document ID.
4. `query2docids`: Returns a list of document IDs for the given corpus and free-text query.

We are still deploying the model on the production server, and will release the `base_url` soon.

### query2vector
```python
import requests

url = f"http://{base_url}/query2vector"
params = {"query": "diabetes and CNS"}
response = requests.get(url, params=params)
vector = response.json()
```

### doc2vector
```python
import requests

url = f"http://{base_url}/doc2vector"
params = {"title": "Diagnosis and Management of Central Diabetes Insipidus in Adults", "text": "Central diabetes insipidus (CDI) is a clinical syndrome which results from loss or impaired function of vasopressinergic neurons in the hypothalamus/posterior pituitary, resulting in impaired synthesis and/or secretion of arginine vasopressin (AVP). AVP deficiency leads to the inability to concentrate urine and excessive renal water losses, resulting in a clinical syndrome of hypotonic polyuria with compensatory thirst. CDI is caused by diverse etiologies, although it typically develops due to neoplastic, traumatic, or autoimmune destruction of AVP-synthesizing/secreting neurons. This review focuses on the diagnosis and management of CDI, providing insights into the physiological disturbances underpinning the syndrome. Recent developments in diagnostic techniques, particularly the development of the copeptin assay, have improved accuracy and acceptability of the diagnostic approach to the hypotonic polyuria syndrome. We discuss the management of CDI with particular emphasis on management of fluid intake and pharmacological replacement of AVP. Specific clinical syndromes such as adipsic diabetes insipidus and diabetes insipidus in pregnancy as well as management of the perioperative patient with diabetes insipidus are also discussed."}
response = requests.get(url, params=params)
vector = response.json()
```

### docid2vector
```python
import requests

url = f"http://{base_url}/docid2vector"
params = {"corpus": "pubmed", "docid": "35771962"}
response = requests.get(url, params=params)
vector = response.json()
```

### query2docids
```python
import requests

url = f"http://{base_url}/query2docids"
params = {"corpus": "pubmed", "query": "What are the relations between lead and heart damage?"}
response = requests.get(url, params=params)
pmids = response.json()
```

## MedCPT weights

If you want to access the MedCPT model weights, please fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSdtd2OmkI3ep_RadoiTxdVSqvR7rSDssDlAyrpQDaddhv5oOw/viewform?usp=sf_link).


## Data availability

Due to [privacy](https://www.nlm.nih.gov/web_policies.html#privacy_security) concerns, we are not able to release the PubMed user logs. As a surrogate, we provide the question-article pair data from [BioASQ](http://www.bioasq.org/) in this repo as example training datasets. You can convert your data to the example data formats and train the MedCPT model.

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.

## Citation

If you find this repo helpful, please cite MedCPT by:

```bibtext
@misc{jin2023MedCPT,
      title={MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval}, 
      author={Qiao Jin and Won Kim and Qingyu Chen and Donald C. Comeau and Lana Yeganova and John Wilbur and Zhiyong Lu},
      year={2023},
      eprint={2307.00589},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
