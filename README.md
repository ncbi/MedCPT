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
- MedCPT weights.
- Code for training the MedCPT [retriever](./retriever/).
- Code for training the MedCPT [re-ranker](./reranker/).
- Code for evaluating the pre-trained model at [evals](./evals/).

## MedCPT weights

MedCPT model weights are publicly available on Hugging Face:
- [MedCPT Query Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder)
- [MedCPT Article Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder)

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
