# Depression Prediction Model for Students - ML Zoomcamp 2024 (MidT Term)

## Overview

Depression Prediction Model project, developed as part of ML Zoomcamp 2024 midterm project. This project focuses on predicting the likelihood of depression among students using machine learning techniques. The model leverages a comprehensive dataset to analyze whether the student is depressed or not given the features.

This model is a classification system designed to predict whether a student is experiencing depression or not based on specific input features. By analyzing various factors related to demographics, academics, lifestyle, and health, **the model assigns each student a binary label**:

- Depressed (1)
- Not Depressed (0)

### What the Model Solves  
The model addresses the **early detection of mental health issues** among students by providing insights into their mental state. This solves several problems:  

#### **Mental Health Awareness**  
- Identifies students who might need support or intervention, enabling targeted help.  

#### **Preventive Measures**  
- Helps schools, universities, and mental health organizations implement early prevention strategies.  

#### **Resource Allocation**  
- Allows institutions to focus mental health resources on at-risk students more efficiently.  

#### **Insights into Risk Factors**  
- Provides a data-driven understanding of factors that correlate strongly with depression, supporting policy-making and student wellness programs.  

### Potential Use Cases  

#### **Counseling Services**  
- Early flagging of students who may require professional mental health support.  

#### **Educational Institutions**  
- Insights from the model can inform institutional policies aimed at reducing academic pressure or enhancing support systems.  

#### **Public Health Studies**  
- The findings could contribute to broader research on mental health trends in student populations.  



## Dataset Description

The model uses the [Depression Student Dataset Prediction](https://www.kaggle.com/code/sameerk2004/depression-student-dataset-prediction) from Kaggle. This dataset explores the correlation between mental health and factors such as:

- **Demographic Information:** Age, gender.
- **Academic Factors:** Academic pressure, study satisfaction, study hours.
- **Lifestyle Factors:** Sleep duration, dietary habits.
- **Economic Factors:** Financial stress.
- **Health Background:** Family history of mental illness.

### Dataset Size
Given the dataset's small size, with approximately 500 rows, extensive model comparison (like decision trees, XGBoost, etc.) was deemed unnecessary, leading me to use basic linearregression() on initial performance metrics.


The dataset serves to identify risk factors for depression, providing insights that could be vital for educational institutions to implement preventive mental health strategies.

## Model Performance

The trained model has achieved the following metrics:

- **Precision:** 0.9803921568627451
- **Recall:** 0.9803921568627451
- **AUC (Area Under Curve):** 0.9801960784313726

These metrics indicate a high level of accuracy in predicting whether a student might be suffering from depression based on the input features.

## How to Use the Model

### Prerequisites

- **Pipenv:** For managing Python environments and dependencies.
- **Docker:** For containerized deployment of the model.
- **Flask:** For app/webservice.

### Running Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/izaanz/ml-zoomcamp-2024/tree/main/Mid%20Term
   navigate to the cloned directory
   ```

2. **Setup Environment:**
   ```bash
   pipenv install
   pipenv shell
   ```

3. **Run the Model:**
   
   Note: You may have to run `python train.py` if the model_.bin doesn't validate on your end.
   
   ```bash
   python predict.py
   ```
   This will start a local server where you can send requests to get predictions.

### Docker Deployment

To deploy using Docker:

1. **Build the Docker Image:**
   ```bash
   docker build -t depression-predictor .
   ```

2. **Run the Docker Container:**
   ```bash
   docker run -p 9696:9696 depression-predictor
   ```

   The model will be accessible at `http://localhost:9696/predict`.

This Dockerfile sets up a Python 3.11 environment, installs Pipenv, and copies the required files into the container. It then exposes port 9696 and sets up the Waitress server to serve the model.

### Interacting with the Model

When testing the model, you can use the following JSON structure for a student's data:


```json
{
  "gender": "male",
  "age": 24,
  "academic_pressure": 2.0,
  "study_satisfaction": 4.0,
  "sleep_duration": "5-6 hours",
  "dietary_habits": "unhealthy",
  "suicidal_thoughts": 1,
  "study_hours": 10,
  "financial_stress": 3,
  "family_history_of_mental_illness": 0
}
```
Use predict_test.py to send test queries to your model:

- Use `predict_test.py` to send test queries to your model:
  ```bash
  python predict_test.py
  ```

  Modify this script to format your input data as per the model's expectations.

### Deployment on Server

The model is currently deployed and accessible at:

- **Server:** `135.181.46.254`
- **Port:** `9696`
- **Endpoint:** `/predict`

- `http://135.181.46.254:9696/predict`

You can interact with this deployment using HTTP POST requests to the endpoint with JSON data formatted according to the dataset features.

## Contributions

Contributions to improve the model, enhance feature sets, or optimize the deployment process are welcome. Please submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to Kaggle for providing the dataset.

---

This README provides a comprehensive guide to understanding, using, and deploying the Depression Prediction Model. For any issues or further information, feel free to open an issue in this repository.
