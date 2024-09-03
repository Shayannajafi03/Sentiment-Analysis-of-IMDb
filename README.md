# IMDB Movie Reviews Analysis

## Overview

This notebook contains the analysis of the IMDB Dataset of 50,000 Movie Reviews. The purpose of this analysis is to explore the dataset, perform sentiment analysis, and extract meaningful insights from movie reviews.

## Dataset

The dataset used for this analysis is sourced from Kaggle. It contains 50,000 movie reviews labeled as either positive or negative. You can download the dataset from the following link:

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### Dataset Details:

- **Number of reviews**: 50,000
- **Labels**: Positive, Negative
- **Columns**: `review`, `sentiment`

## Requirements

To run this notebook, you need the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

## Notebook Contents

1. **Data Loading and Preprocessing**:
   - Load the dataset and preprocess the text data by removing special characters, stopwords, and performing tokenization.

2. **Exploratory Data Analysis (EDA)**:
   - Analyze the distribution of sentiments and visualize the most common words in positive and negative reviews.

3. **Sentiment Analysis**:
   - Implement machine learning models such as Logistic Regression and Naive Bayes to classify the movie reviews as positive or negative.
   - Evaluate the performance of the models using metrics like accuracy, precision, recall, and F1-score.

4. **Conclusion**:
   - Summarize the findings and provide insights based on the analysis.

## How to Use

1. Download the dataset from the link provided above.
2. Place the dataset file in the same directory as the notebook or provide the correct path in the notebook.
3. Run the cells in the notebook to perform the analysis.

## Acknowledgments

- The dataset is provided by Lakshmi Narayanan on Kaggle.
- This notebook utilizes various open-source libraries, and their respective contributors are acknowledged.
