# DeepLearningProject
This is a semester project of deep learning course. We applied transformers models on NLP domain (NER) and reported results.

# Transformer Models Implementation
This repository contains Jupyter Notebook files implementing various Transformer models for natural language processing tasks using libraries such as PyTorch and Hugging Face Transformers.

## Overview
The Transformer architecture has revolutionized the field of natural language processing (NLP) with its ability to capture long-range dependencies and learn contextual representations effectively. In this repository, we provide implementations of popular Transformer models such as BERT, GPT, and T5, along with examples demonstrating their usage for tasks like text classification, language generation, and more.

## Problem Statement
Automated writing feedback tools offer a potential solution to the low proficiency in writing among various students. However, current tools have limitations and are often inaccessible to educators due to cost. We aims to address these issues by developing innovative tools for social good by identifying elements in student writing, including segmenting texts and classifying argumentative and rhetorical elements in essays from given dataset (feedback writing) of students. 

## Dataset 
The dataset comprises argumentative essays from U.S. students in grades 6-12, annotated for elements commonly found in argumentative writing. We predicted human annotations by segmenting essays into rhetorical and argumentative elements and classifying them into categories such as Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, and Concluding Statement. The training set includes individual essays in .txt format and a .csv file containing annotations. The test set consists of approximately 15k documents in .txt format, and submissions should follow a specific format outlined in the sample_submission.csv file.

## Methodology
We utilized a Transformer model for processing the argumentative essays dataset. The model was trained using a custom loss function, `CustomNonPaddingTokenLoss`, which applies a sparse categorical cross-entropy loss to non-padding tokens in the sequences. The Transformer architecture consists of `TransformerBlock` layers, each containing a multi-head self-attention mechanism and a feed-forward neural network (FFN). Layer normalization and dropout were applied to stabilize training and prevent overfitting. Additionally, positional embeddings were incorporated using a `PositionalEmbedding` layer to provide the model with information about the order of tokens in the input sequences. These components collectively enabled the model to effectively learn the structural and contextual features of the essays for the task of segmenting and classifying discourse elements.

## Repository Structure
- **Notebooks**: This directory contains Jupyter Notebook files (.ipynb) with implementations of Transformer models and examples of how to use them for different NLP tasks.
  
- **Data**: Here, you can find sample datasets used in the notebooks for training and evaluation purposes.

## Requirements
To run the notebooks in this repository, you need to have the following installed:

- Python 3.x
- Jupyter Notebook
- PyTorch
- Hugging Face Transformers

You can install these dependencies using pip:

```bash
pip install torch
pip install transformers
```

## Usage
1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/transformer-models.git
```

2. Navigate to the cloned directory:

```bash
cd transformer-models
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the desired notebook from the `Notebooks` directory and run the cells to see the implementation and results.
5. 
## Contributing
Contributions are welcome! If you have suggestions, bug reports, or want to add new features, feel free to open an issue or submit a pull request.
