# BioCPT

This repo contains the code for training the BioCPT model. BioCPT is a first-of-its-kind Contrastive Pre-trained Transformer model trained with an unprecedented scale of PubMed search logs for zero-shot biomedical information retrieval. 

BioCPT contains:
- A frist-stage dense retriever, which is trained by 255M query-article pairs from PubMed search logs and in-batch negatives. The BioCPT retriever contains a query encoder and an article encoder, both of which are initialized by PubMedBERT.   
- A second-stage re-ranker, which is trained by 18M semantic query-article pairs and localized negativesfrom the pre-trained dense retriver. The re-ranker is a transformer cross-encoder that is initialized by PubMedBERT.

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
