# Crop Yield Prediction Using Machine Learning

This repository contains a machine learning project for predicting crop yield based on various features such as year, average rainfall, pesticide usage, average temperature, area, and crop type.

## Project Structure

## Files and Directories

- `Application/`: Contains the Flask web application.
  - `app.py`: The main Flask application file.
  - `CropYeild.pkl`: The trained crop yield prediction model.
  - `Preprocessor.pkl`: The preprocessor used for transforming input features.
  - `static/`: Directory for static files (e.g., images, CSS).
  - `templates/`: Directory for HTML templates.
    - `index.html`: The main HTML template for the web application.
- `Crop Yeild Prediction Using ML.ipynb`: Jupyter notebook used for data analysis, preprocessing, and model training.
- `Dataset/`: Contains the datasets used for training the model.
  - `pesticides.csv`
  - `rainfall.csv`
  - `temp.csv`
  - `yield_df.csv`
  - `yield.csv`
- `README.md`: This file.
- `requirements.txt`: List of Python dependencies required for the project.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/winter000boy/Crop-Yeild-Prediction.git
   cd crop-yield-prediction


2. Create a virtual environment and activate it:
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate`


3. Install the required dependencies:
    pip install -r requirements.txt



## Usage

1. Run the Flask application:
cd Application
python app.py

2. Open your web browser and go to http://127.0.0.1:5000/ to access the crop yield prediction application.

## Data Preprocessing and Model Training
The Jupyter notebook Crop Yeild Prediction Using ML.ipynb contains the code for data preprocessing, exploratory data analysis, and model training. It includes the following steps:

Importing necessary libraries.
Loading and exploring the dataset.
Data cleaning and preprocessing.
Feature engineering.
Model training and evaluation.
Saving the trained model and preprocessor using pickle.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.



License
This project is licensed under the MIT License. See the LICENSE file for details.

https://github.com/winter000boy/Crop-Yeild-Prediction.git
