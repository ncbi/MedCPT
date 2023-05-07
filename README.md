# BioCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval

## Overview

![image](https://user-images.githubusercontent.com/32558774/236640954-bfa0d9da-50b5-43b3-8326-bf2e3b9f4b33.png)

BioCPT is a first-of-its-kind Contrastive Pre-trained Transformer model trained with an unprecedented scale of PubMed search logs for zero-shot biomedical information retrieval. BioCPT consists of:
- A frist-stage dense retriever (BioCPT retriever)
  - Contains a query encoder (QEnc) and an article encoder (DEnc), both initialized by [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).   
  - Trained by 255M query-article pairs from PubMed search logs and in-batch negatives. 
- A second-stage re-ranker (BioCPT re-ranker)
  - A transformer cross-encoder (CrossEnc) initialized by [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).
  - Trained by 18M semantic query-article pairs and localized negatives from the pre-trained BioCPT retriever. 

## Content

This directory contains:
- Code for training the BioCPT retriever at `./retriever/`.
- Code for training the BioCPT re-ranker at `./reranker/`.

## Data availability

Due to [privacy](https://www.nlm.nih.gov/web_policies.html#privacy_security) concerns, we are not able to release the PubMed user logs. As a surrogate, we provide the question-article pair data from [BioASQ](http://www.bioasq.org/) in this repo as example training datasets.

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
