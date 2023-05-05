# BioCPT
This repo contains the code for training the BioCPT model. BioCPT is a first-of-its-kind Contrastive Pre-trained Transformer model trained with an unprecedented scale of PubMed search logs for zero-shot biomedical information retrieval. 

BioCPT contains:
- A frist-stage dense retriever, which is trained by 255M query-article pairs from PubMed search logs and in-batch negatives. The BioCPT retriever contains a query encoder and an article encoder, both of which are initialized by PubMedBERT.   
- A second-stage re-ranker, which is trained by 18M semantic query-article pairs and localized negativesfrom the pre-trained dense retriver. The re-ranker is a transformer cross-encoder that is initialized by PubMedBERT.
