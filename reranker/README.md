# BioCPT re-ranker

This repo contains the code for training the BioCPT retriever (Part B in the figure below). 
![image](https://user-images.githubusercontent.com/32558774/236641890-aaf42b3f-b114-4da1-87c7-a7d47ae29fbb.png)

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
$ head train_example.json


# qid2info.json is a json dict where keys are qids and values are the queries (BioASQ questions in the example)
$ head qid2info_example.json 


# pmid2info.json is a json dict where keys are pmids and values are a tuple (list) of title and abstract.
$ head pmid2info_example.json 

```

## Training BioCPT on BioASQ
You can directly train the BioCPT retriever with the provided BioASQ datasets by running:
```bash
bash run.sh
```

## Custom usage
You can also use run `main.py` with other arguments or datasets for custom usage:
```bash
$ python main.py --help

```

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
