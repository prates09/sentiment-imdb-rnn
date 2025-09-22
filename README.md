# Sentiment Analysis on IMDB: LSTM vs GRU vs Bi-LSTM

This repository contains the full code, notebook, dataset links, and trained models for the project  
**“Comparative Analysis of LSTM, GRU, and Bi-LSTM on the IMDB Movie-Review Dataset.”**

The Jupyter notebook automatically downloads the dataset, trains the three models, and saves all figures and trained weights used in the research paper.

---

## Dataset
- **IMDB 50k Movie Reviews**  
  • Official site: https://ai.stanford.edu/~amaas/data/sentiment/  
  • Hugging Face dataset card: https://huggingface.co/datasets/imdb  

The notebook loads the dataset directly:
```python
from datasets import load_dataset
imdb = load_dataset("imdb")
