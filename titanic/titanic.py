import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


class Axis():
    row = 0
    col = 1


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

train_x = train.drop(["Survived"], axis=Axis.col)
train_y = train["Survived"]

test_x = test.copy()

# -----------------------------------
# 特徴量作成
# -----------------------------------

train_x = train_x.drop(["PassengerId"], axis=Axis.col)
test_x = test_x.drop(["PassengerId"], axis=Axis.col)

train_x = train_x.drop(["Name", "Ticket", "Cabin"], axis=Axis.col)
test_x = test_x.drop(["Name", "Ticket", "Cabin"], axis=Axis.col)

for category in ["Sex", "Embarked"]:
    le = LabelEncoder()
    le.fit(train_x[category].fillna('NA'))

    train_x[category] = le.transform(train_x[category].fillna('NA'))
    test_x[category] = le.transform(test_x[category].fillna('NA'))

# -----------------------------------
# モデル作成
# -----------------------------------
model = XGBClassifier(n_estimators=20, random_state=42)
model.fit(train_x, train_y)

# [Survived prob, non Survived prob]
pred = model.predict_proba(test_x)
pred_serise = pred[:, 1]

pred_label = np.where(pred_serise > 0.5, 1, 0)

# 提出用ファイルの作成
submission = pd.DataFrame(
    {'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)

# -----------------------------------
# バリデーション
# -----------------------------------

# 各foldのスコアを保存するリスト
score_accuracy = []
score_logloss = []

# クロスバリデーションを行う
# 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    model = XGBClassifier(n_estimators=20, random_state=42)
    model.fit(tr_x, tr_y)

    va_pred = model.predict_proba(va_x)
    va_pred_serise = va_pred[:, 1]

    accuracy = accuracy_score(va_y, va_pred_serise > 0.5)
    logloss = log_loss(va_y, va_pred_serise)

    score_accuracy.append(accuracy)
    score_logloss.append(logloss)

logloss = np.mean(score_logloss)
accuracy = np.mean(score_accuracy)
print(f"logloss:{logloss:.4f}, accuracy:{accuracy:.4f}")

# -----------------------------------
# モデルチューニング
# -----------------------------------

param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

param_combinations = itertools.product(
    param_space['max_depth'], param_space['min_child_weight'])

params = []
scores_logloss = []

for max_depth, min_child_weight in param_combinations:
    score_logloss_folds = []

    # クロスバリデーションを行う
    # 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        model = XGBClassifier(n_estimators=20, random_state=42,
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        va_pred = model.predict_proba(va_x)
        va_pred_serise = va_pred[:, 1]

        logloss = log_loss(va_y, va_pred_serise)

        score_logloss_folds.append(logloss)

    score_mean = np.mean(score_logloss_folds)

    params.append((max_depth, min_child_weight))
    scores_logloss.append(score_mean)

best_score_idx = np.argsort(scores_logloss)[0]
best_param = params[best_score_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')

# -----------------------------------
# ロジスティック回帰用の特徴量の作成
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder
# 元データをコピーする
train_x2 = train.drop(['Survived'], axis=1)
test_x2 = test.copy()

# 変数PassengerIdを除外する
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# 変数Name, Ticket, Cabinを除外する
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# one-hot encodingを行う
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(train_x2[cat_cols].fillna('NA'))

# one-hot encodingのダミー変数の列名を作成する
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# one-hot encodingによる変換を行う
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# one-hot encoding済みの変数を除外する
train_x2 = train_x2.drop(cat_cols, axis=Axis.col)
test_x2 = test_x2.drop(cat_cols, axis=Axis.col)

# one-hot encodingで変換された変数を結合する
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=Axis.col)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=Axis.col)

# 数値変数の欠損値を学習データの平均で埋める
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 変数Fareを対数変換する
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])

# -----------------------------------
# アンサンブル
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboostモデル
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# ロジスティック回帰モデル
# xgboostモデルとは異なる特徴量を入れる必要があるので、別途train_x2, test_x2を作成した
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 予測値の加重平均をとる
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)

submission = pd.DataFrame(
    {'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_second.csv', index=False)