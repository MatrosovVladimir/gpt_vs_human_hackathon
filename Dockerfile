FROM python:3.8
WORKDIR /app
COPY . /app
VOLUME /app/data
RUN pip3 install -r requirements.txt
RUN pip3 install IPython
RUN pip3 install traitlets
RUN python3 -c "import os;import pandas as pd;from catboost import CatBoostClassifier;from catboost import Pool;from sklearn.model_selection import train_test_split;from sentence_transformers import SentenceTransformer;from sklearn import svm; SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1');"
RUN chmod +x /app/gpt-human-catboost.py
CMD ["python3","/app/gpt-human-catboost.py"]
