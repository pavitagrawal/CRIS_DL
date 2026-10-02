# Customer Review Intelligence System

### ICT 4442 – Deep Learning Mini Project

## Overview

This project conducts a comparative study of four deep learning architectures for **Aspect-Based Sentiment Analysis (ABSA)** of Amazon customer reviews.

Unlike traditional sentiment classifiers, the system identifies important product aspects—such as quality, price, design, performance, battery, and customer support—and classifies the sentiment associated with each aspect as positive, negative, or neutral.

## Team Members and Model Assignments

| Team Member | Assigned Model |
|---|---|
| Pavit Agrawal | Multilayer Perceptron (MLP) |
| Priyanshu Sharma | TextCNN |
| Shivam Lahoty | Bidirectional LSTM (BiLSTM) |
| Aryan Sorout | Attention-Based Transformer and Dashboard |

## Dataset

**Amazon Customer Reviews Dataset** — McAuley Lab, UCSD

- Categories: Electronics, Beauty, Books, and Home & Kitchen
- Fields: Review text, review summary, star rating (1–5), product ID (ASIN), and category
- Dataset split: 70% training, 15% validation, and 15% testing
- Splitting method: Stratified sampling

## Project Structure

```text
CustomerReview-SentimentAnalysis/
├── data/                         # Raw and processed datasets
├── models/                       # Model architecture scripts
│   ├── model_mlp.py              # MLP model (Pavit Agrawal)
│   ├── model_textcnn.py          # TextCNN model (Priyanshu Sharma)
│   ├── model_bilstm.py           # BiLSTM model (Shivam Lahoty)
│   └── model_transformer.py      # Transformer model (Aryan Sorout)
├── notebooks/                    # Jupyter notebooks for experiments
├── results/                      # Metrics, confusion matrices, and plots
├── utils/                        # Shared utility functions
│   └── preprocessing.py          # Data preprocessing pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Evaluation Protocol

All four models use the same training, validation, and testing datasets to ensure a fair comparison.

The following evaluation metrics are used:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score
- Confusion Matrix
- Training Time
- Prediction Time

## References

1. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP.
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
3. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
4. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
5. Pontiki, M., et al. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*.
6. Zhang, L., Wang, S., & Liu, B. (2018). *Deep Learning for Sentiment Analysis: A Survey*. WIREs Data Mining and Knowledge Discovery.