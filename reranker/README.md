# MedCPT re-ranker

This repo contains the code for training the MedCPT re-ranker (Part B in the figure below). 
![image](https://user-images.githubusercontent.com/32558774/236641890-aaf42b3f-b114-4da1-87c7-a7d47ae29fbb.png)

## Requirements

The code has been tested with Python 3.9.13. Please first instlal the required packages by:
```bash
pip install -r requirements.txt
```

## Training datasets

We provide the [BioASQ](http://www.bioasq.org/) question-article pairs at `./datasets/` as the training datasets for demonstration. Due to privacy concerns, we are not able to release the user logs of PubMed.
Generally, the MedCPT re-ranker requires three files for training: `training.json`, `qid2info.json`, and `pmid2info.json`.  You can convert your data to the example data formats and train the MedCPT model. Their formats are shown below:
```bash
# train.jsonl is a json list file where each entry contains a dict of Dict{'qid': Str(qid), 'pos_pmids': List[List[Str(pmid), Int(click)]], 'neg_pmids': List[Str(pmid)]}
$ head train_example.json
[
    {
        "qid": "0",
        "pos_pmids": [
            [
                "15858239",
                1
            ],
            [
                "15829955",


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
    "6650562": [
```

## Training MedCPT on BioASQ
You can directly train the MedCPT re-ranker with the provided BioASQ datasets by running:
```bash
bash run.sh
```

## Custom usage
You can also use run `main.py` with other arguments or datasets for custom usage:
```bash
$ python main.py --help
usage: main.py [-h] [--bert_path BERT_PATH] [--tokenizer_path TOKENIZER_PATH] --output_dir OUTPUT_DIR
               [--pmid2info_path PMID2INFO_PATH] [--qid2info_path QID2INFO_PATH] [--train_dataset TRAIN_DATASET]
               [--max_query_length MAX_QUERY_LENGTH] [--max_doc_length MAX_DOC_LENGTH] [--learning_rate LEARNING_RATE]
               [--num_neg_pmids NUM_NEG_PMIDS] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
               [--num_train_epochs NUM_TRAIN_EPOCHS] [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON]
               [--max_grad_norm MAX_GRAD_NORM] [--warmup_steps WARMUP_STEPS] [--logging_steps LOGGING_STEPS]
               [--save_steps SAVE_STEPS] [--eval_all_checkpoints] [--no_cuda] [--overwrite_cache]
               [--do_lower_case DO_LOWER_CASE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --bert_path BERT_PATH
                        The path of the pre-trained query encoder.
  --tokenizer_path TOKENIZER_PATH
                        The path of the tokenizer.
  --output_dir OUTPUT_DIR
                        The output directory where the model checkpoints and predictions will be written.
  --pmid2info_path PMID2INFO_PATH
                        The path to pmid2info json file.
  --qid2info_path QID2INFO_PATH
                        The path to qid2info json file.
  --train_dataset TRAIN_DATASET
                        The path of the training dataset.
  --max_query_length MAX_QUERY_LENGTH
                        Max length of query.
  --max_doc_length MAX_DOC_LENGTH
                        Max length of documents.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --num_neg_pmids NUM_NEG_PMIDS
                        Negative pmids per batch
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --eval_all_checkpoints
                        Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step
                        number
  --no_cuda             Whether not to use CUDA when available
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --do_lower_case DO_LOWER_CASE
                        Set this flag if you are using an uncased model. Queries are uncased, so setting default to
                        True..
  --seed SEED           random seed for initialization
```

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
