# 📰 fake-news-detection-system
A machine learning project that detects fake news articles using natural language processing techniques. This system aims to classify news articles as fake or real. The project utilizes various machine learning models to achieve high accuracy.

## Tech Stack
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![HTML](https://img.shields.io/badge/HTML-5-orange)](https://www.w3schools.com/html/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-green)](https://scikit-learn.org/)

## Features
* Detects fake news articles using natural language processing techniques
* Utilizes multiple machine learning models for high accuracy
* Provides a simple web interface for users to input news articles
* Displays the classification result (fake or real) for the input news article

## Folder Structure
```
fake-news-detection-system
├── app.py
├── data
│   ├── Fake.csv
│   ├── Real.csv
├── models
│   ├── all_models.pkl
├── templates
│   ├── index.html
│   ├── result.html
├── train.py
```

## How to Run Locally
1. Clone the repository using `git clone https://github.com/your-username/fake-news-detection-system.git`
2. Navigate to the project directory using `cd fake-news-detection-system`
3. Install the required dependencies using `pip install -r requirements.txt` (create a requirements.txt file with necessary libraries like scikit-learn, flask, etc.)
4. Train the machine learning models using `python train.py`
5. Run the web application using `python app.py`
6. Open a web browser and navigate to `http://localhost:5000` to use the fake news detection system

## Contributing Guide
To contribute to this project, fork the repository, make the necessary changes, and submit a pull request. Ensure that your changes are well-documented and follow standard professional guidelines.