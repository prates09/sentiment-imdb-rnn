# Sentiment Analysis on IMDB: LSTM vs GRU vs Bi-LSTM

This repository contains the full code, notebook, dataset links, and trained models for the project  
**“Comparative Analysis of LSTM, GRU, and Bi-LSTM on the IMDB Movie-Review Dataset.”**

The Jupyter notebook automatically downloads the dataset, trains the three models, and saves all figures and trained weights used in the research paper.

---

## Dataset
- **IMDB 50k Movie Reviews**  
  • Official site: https://ai.stanford.edu/~amaas/data/sentiment/  
  • Hugging Face dataset card: https://huggingface.co/datasets/imdb  
  ## Preprocessing

- Tokenization: Splitting text into words.
- Vocabulary: Limited to the top 10,000 most frequent words.
- Sequence Length: Each review truncated/padded to 200 tokens.
- Embeddings: 128-dimensional word embeddings.

## Model Architectures

- **LSTM**: Embedding → 64-unit LSTM → Dense layer.
- **GRU**: Embedding → 64-unit GRU → Dense layer.
- **BiLSTM**: Embedding → 64-unit bidirectional LSTM → Dense layer.

All models trained with Adam optimizer (lr=0.001), binary cross-entropy loss, batch size 64, up to 15 epochs with early stopping.

## Results

### Overall Metrics
| Model   | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| LSTM    | 0.861    | 0.862     | 0.861  | 0.861    |
| GRU     | 0.861    | 0.840     | 0.892  | 0.865    |
| BiLSTM  | 0.859    | 0.905     | 0.803  | 0.851    |

- LSTM: Balanced performance.  
- GRU: Best recall (captures more positives).  
- BiLSTM: Best precision (fewer false positives).  

**Figure 1:** Bar chart comparison of Accuracy, Precision, Recall, and F1.  
**Figure 2:** Training vs Validation Accuracy curves.  
**Figure 3a–c:** Confusion matrices for LSTM, GRU, BiLSTM.

## Discussion

- **Accuracy:** All models hover around 86%, showing strong performance on IMDB reviews.  
- **Recall:** GRU reduces false negatives, useful when capturing positive sentiment is critical (e.g., customer satisfaction).  
- **Precision:** BiLSTM minimizes false positives, suitable for applications requiring high reliability (e.g., toxic content detection).  
- **Efficiency:** GRU trains faster due to fewer parameters; BiLSTM is computationally heavier. 

## Future Work

- Extend experiments to larger datasets (Amazon, Twitter, multilingual).  
- Explore pretrained embeddings (FastText, GloVe) or transformer models (BERT).  
- Hyper-parameter tuning for batch size, sequence length, and embedding dimensions.  
- Deployment as an API or web app for real-time sentiment analysis.  

## How to Run

Clone this repo and install dependencies:
```bash
git clone https://github.com/yourusername/IMDB-Sentiment-RNN.git
cd IMDB-Sentiment-RNN
pip install -r requirements.txt





The notebook loads the dataset directly:
```python
from datasets import load_dataset
imdb = load_dataset("imdb")

