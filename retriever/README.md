# BioCPT retriever

This repo contains the code for training the BioCPT retriever. 

## Requirements

The code has been tested with Python 3.9.13. Please first instlal the required packages by:
```bash
pip install -r requirements.txt
```

## Training datasets

We provide the [BioASQ](http://www.bioasq.org/) question-article pairs at `./datasets/` as the training datasets for demonstration. Due to privacy concerns, we are not able to release the user logs of PubMed.
Generally, the BioCPT retriever requires three files for training: `training.jsonl`, `qid2info.json`, and `pmid2info.json`. Their formats are shown below:
```bash
# train.jsonl is a jsonline file where each line contains a json of query-article article and the number of click
$ head train_example.jsonl 
{"qid": "0", "pmid": "15858239", "click": 1}
{"qid": "0", "pmid": "15829955", "click": 1}
{"qid": "0", "pmid": "6650562", "click": 1}
{"qid": "0", "pmid": "12239580", "click": 1}
{"qid": "0", "pmid": "21995290", "click": 1}
{"qid": "0", "pmid": "23001136", "click": 1}
{"qid": "0", "pmid": "15617541", "click": 1}
{"qid": "0", "pmid": "8896569", "click": 1}
{"qid": "0", "pmid": "20598273", "click": 1}
{"qid": "1", "pmid": "23959273", "click": 1}

# qid2info.json is a json dict where keys are qids and values are the queries (BioASQ questions in the example)
$ head qid2info_example.json 
{
    "0": "Is Hirschsprung disease a mendelian or a multifactorial disorder?",
    "1": "List signaling molecules (ligands) that interact with the receptor EGFR?",
    "2": "Is the protein Papilin secreted?",
    "3": "Are long non coding RNAs spliced?",
    "4": "Is RANKL secreted from the cells?",
    "5": "Does metformin interfere thyroxine absorption?",
    "6": "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?",
    "7": "Which acetylcholinesterase inhibitors are used for treatment of myasthenia gravis?",
    "8": "Has Denosumab (Prolia) been approved by FDA?",

# pmid2info.json is a json dict where keys are pmids and values are a tuple (list) of title and abstract.
$ head pmid2info_example.json 
{
    "15858239": [
        "[The role of ret gene in the pathogenesis of Hirschsprung disease].",
        "Hirschsprung disease is a congenital disorder with the incidence of 1 per 5000 live births, characterized by the absence of intestinal ganglion cells. In the etiology of Hirschsprung disease various genes play a role; these are: RET, EDNRB, GDNF, EDN3 and SOX10, NTN3, ECE1, Mutations in these genes may result in dominant, recessive or multifactorial patterns of inheritance. Diverse models of inheritance, co-existence of numerous genetic disorders and detection of numerous chromosomal aberrations together with involvement of various genes confirm the genetic heterogeneity of Hirschsprung disease. Hirschsprung disease might well serve as a model for many complex disorders in which the search for responsible genes has only just been initiated. It seems that the most important role in its genetic etiology plays the RET gene, which is involved in the etiology of at least four diseases. This review focuses on recent advances of the importance of RET gene in the etiology of Hirschsprung disease."
    ],
    "15829955": [
        "A common sex-dependent mutation in a RET enhancer underlies Hirschsprung disease risk.",
        "The identification of common variants that contribute to the genesis of human inherited disorders remains a significant challenge. Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes. We have used family-based association studies to identify a disease interval, and integrated this with comparative and functional genomic analysis to prioritize conserved and functional elements within which mutations can be sought. We now show that a common non-coding RET variant within a conserved enhancer-like sequence in intron 1 is significantly associated with HSCR susceptibility and makes a 20-fold greater contribution to risk than rare alleles do. This mutation reduces in vitro enhancer activity markedly, has low penetrance, has different genetic effects in males and females, and explains several features of the complex inheritance pattern of HSCR. Thus, common low-penetrance variants, identified by association studies, can underlie both common and rare diseases."
    ],
```

## Training BioCPT on BioASQ
You can directly train the BioCPT retriever with the provided BioASQ datasets by running:
```bash
bash run.sh
```

## Custom usage



