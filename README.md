# InsightLyzer - Ml Video Analysis

## Overview

InsightLyzer is a powerful web application that enables users to analyze video of big datasets using various machine learning algorithms. The application provides comprehensive tools for data exploration, model training, and visualization of results. Dataset must include likes and dislikes columns.

## Key Features

- **Dataset Analysis**:
  - Upload and explore CSV/Excel datasets
  - After upload and analyze you can go Result tab
  - View descriptive statistics and data types
  - Handle large datasets with smart sampling

- **Machine Learning**:
  - Multiple algorithm options:
    - K-Nearest Neighbors (KNN)
    - K-Means Clustering
    - Naive Bayes
    - Logistic Regression
    - Decision Trees
    - Support Vector Machines (SVM)
  - Customizable test/train split
  - Performance metrics (accuracy, confusion matrix)

- **Video Analytics**:
  - Top 10 most liked videos
  - Top 10 most disliked videos
  - Unique video analysis (non-overlapping liked/disliked)
  - Like/Dislike ratio metrics

- **Visualizations**:
  - Interactive charts (bar, pie, histogram, scatter)
  - Comparative analysis views
  - Customizable chart types

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Machine Learning**: scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/insightlyzer.git
   cd insightlyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the application at:
   ```
   http://localhost:5000
   ```

## Usage

1. Upload your dataset (CSV or Excel format)
2. Explore dataset statistics and structure
3. Select target and feature columns
4. Choose machine learning algorithm
5. View analysis results and visualizations
6. Explore top video metrics

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

**Note**: This is a prototype application. For production use, additional security measures and optimizations would be required.
