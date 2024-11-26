FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--port=9696", "predict:app"]