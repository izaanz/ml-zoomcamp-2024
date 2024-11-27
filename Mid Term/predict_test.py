#!/usr/bin/env python
# coding: utf-8

import requests


host = "135.181.46.254"
url = f'http://{host}:9696/predict'

student = {
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


response = requests.post(url, json=student).json()

print(f"Depression Probability: {response['depression_probability']}\nIs Depressed: {response['is_depressed']}")


if response['is_depressed'] == True:
    print("The Student is depresssed")
else:
    print("The Student isn't depressed")




