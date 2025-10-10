# Mental Health Sentiment Analysis

A comprehensive machine learning project for analyzing sentiment in mental health-related text data using natural language processing and various classification algorithms.

## Project Overview

This project focuses on sentiment classification of mental health statements using advanced NLP techniques and machine learning models. The analysis includes data preprocessing, exploratory data analysis, feature engineering, and multiple classification approaches to identify different mental health conditions from text data.

## Features

- **Data Preprocessing**: Text cleaning, tokenization and lemmatization
- **Sentiment Analysis**: VADER sentiment scoring for emotional analysis
- **Feature Engineering**: TF-IDF vectorization, word length, and sentiment features
- **Machine Learning Models**: 
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- **Clustering Analysis**: K-Means clustering for pattern discovery
- **Visualization**: Word clouds, confusion matrices, and feature importance plots
- **Class Imbalance Handling**: SMOTE for synthetic minority oversampling

## Dataset

The project uses a mental health dataset containing statements labeled with various mental health conditions:
- Anxiety
- Bipolar
- Depression
- Normal
- Personality disorder
- Stress
- Suicidal

**Data Source**: [Sentiment Analysis for Mental Health Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) by Suchintika Sarkar on Kaggle.

## Installation

### Prerequisite

1. Conda environment manager

### Setup

1. Clone or download this repository
2. Navigate to the project directory
3. Create virtual environment ('mental_health_analysis_env') for this project and install required dependencies:

```bash
conda env create -f environment.yml
```

## Usage

### Running the Analysis

1. Ensure your data file is placed in the `data/` directory as `Combined Data.csv`
2. Open the Jupyter notebook:
3. Select the following Kernel:
   'mental_health_analysis_env'
4. Run all cells to execute the complete analysis pipeline

### Project Structure

```
├── data/
│   ├── Combined Data.csv                   # Main dataset
├── mental_health_sentiment_analysis.ipynb  # Main analysis notebook
├── environment.yml                         # Python dependencies
└── README.md                               # This file
```

## Methodology

### 1. Data Preparation
- Load and split data into train/test sets (70%/30%)
- Handle missing values and data quality issues

### 2. Text Preprocessing
- Remove URLs, handles, and special characters
- Convert to lowercase and tokenize
- Remove stop words
- Apply lemmatization

### 3. Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **Sentiment Scores**: VADER sentiment analysis (negative, neutral, positive, compound)
- **Text Length**: Word count as a feature
- **Important Words**: Filter words based on VADER sentiment lexicon

### 4. Machine Learning
- **Classification**: Multiple algorithms for mental health status prediction
- **Clustering**: K-Means clustering for pattern discovery
- **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrices

### 5. Visualization
- Word clouds for different mental health categories
- Sentiment score distributions
- Feature importance analysis
- Clustering visualizations

## Results

The project achieves the following performance metrics:

- **Logistic Regression**: 73.35% accuracy
- **Random Forest**: 75.32% accuracy  
- **XGBoost**: Best overall performance with balanced precision and recall

## Dependencies

### Core Libraries
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms
- `xgboost`: Gradient boosting framework

### NLP Libraries
- `nltk`: Natural language processing toolkit
- `vaderSentiment`: Sentiment analysis

### Visualization
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `wordcloud`: Word cloud generation

### Data Processing
- `imbalanced-learn`: Handling class imbalance
- `scipy`: Scientific computing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with data privacy regulations when working with mental health data.

