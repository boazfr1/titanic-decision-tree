# Titanic Survival Prediction using Decision Tree

A machine learning project that predicts passenger survival on the Titanic using a Decision Tree classifier. This project is part of the practical Deep Learning for Coders learning journey.

## Overview

This project uses the famous Titanic dataset from Kaggle to predict whether passengers survived the disaster. The model analyzes passenger features like age, fare, class, sex, and embarkation port to make predictions.

## Features

- **Data Download**: Automatically downloads the Titanic dataset from Kaggle
- **Data Processing**: Handles missing values and feature engineering
- **Model Training**: Uses Decision Tree classifier with optimized parameters
- **Validation**: Includes train/validation split for model evaluation

## Dataset

The project uses the Titanic dataset which includes:
- **train.csv**: Training data with survival labels
- **test.csv**: Test data for predictions
- **gender_submission.csv**: Sample submission format

### Key Features Used
- **Continuous Variables**: Age, SibSp (siblings/spouses), Parch (parents/children), LogFare, Pclass
- **Categorical Variables**: Sex, Embarked (port of embarkation)
- **Target Variable**: Survived (0 = No, 1 = Yes)

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### Kaggle API Configuration
To download the dataset automatically, you need to set up Kaggle API credentials:

1. Go to your Kaggle account settings
2. Create a new API token (downloads `kaggle.json`)
3. Place the file in `~/.kaggle/kaggle.json`
4. Or update the `creds` variable in `main.py` with your credentials

## Usage

Run the main script:

```bash
chmod +x ./setup_and_run.sh
./setup_and_run.sh
```

The script will:
1. Download the Titanic dataset (if not already present)
2. Process and clean the data
3. Split data into training and validation sets
4. Train a Decision Tree classifier
5. Evaluate the model and print the Mean Absolute Error

## Model Details

- **Algorithm**: Decision Tree Classifier
- **Max Leaf Nodes**: 55 (optimized parameter)
- **Validation Split**: 75% training, 25% validation
- **Random Seed**: 42 (for reproducibility)

## Data Processing Steps

1. **Missing Value Handling**: Fills missing values with mode for categorical variables
2. **Fare Processing**: Fills missing fares with 0 and creates LogFare feature
3. **Categorical Encoding**: Converts categorical variables to numerical codes
4. **Feature Engineering**: Creates logarithmic transformation of fare prices

## Project Structure

```
titanic-using-random-forest/
├── main.py              # Main script with data processing and model training
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── titanic/            # Dataset directory (created automatically)
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
└── titanic.zip         # Downloaded dataset archive
```

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- kaggle >= 1.5.0

## Results

The model outputs the Mean Absolute Error on the validation set, providing a measure of prediction accuracy.

## Learning Objectives

This project demonstrates:
- Data preprocessing and cleaning techniques
- Handling missing values in real-world datasets
- Feature engineering (logarithmic transformations)
- Train/validation split methodology
- Decision Tree classification
- Model evaluation using Mean Absolute Error

## Next Steps

Potential improvements and extensions:
- Try Random Forest classifier for better performance
- Implement cross-validation
- Feature selection and importance analysis
- Hyperparameter tuning
- Generate predictions for the test set
- Create submission file for Kaggle competition

## License

This project is for educational purposes as part of the practical Deep Learning for Coders course.
