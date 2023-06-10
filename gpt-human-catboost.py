import pandas as pd
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool


df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

y_train = df_train["label"].map({'ai_answer': 1, 'hu_answer': 0})

train_pool = Pool(
    data=df_train[['ans_text', 'q_title']],
    label=y_train,
    text_features=['ans_text', 'q_title']
)

model = CatBoostClassifier(iterations=2500, learning_rate=0.115, eval_metric='F1')

model.fit(
    train_pool,
    verbose=False,
    early_stopping_rounds=50,
    metric_period=10)

y_pred = model.predict(df_test[['ans_text', 'q_title']])

df_test["label"] = y_pred
df_test["label"] = df_test["label"].map({1: 'ai_answer', 0: 'hu_answer'})
df_test[["line_id", "label"]].to_csv("data/submission.csv", sep=",", index=False)


