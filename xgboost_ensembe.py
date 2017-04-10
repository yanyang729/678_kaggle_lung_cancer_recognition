import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import gmean
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from glob2 import glob

data = np.load("../muchdata/muchdata.npy",encoding='bytes')
x, y = [], []
for i, j in data:
    x.append(i)
    y.append(j)

x = np.array([c.flatten() for c in x])
y = np.array([np.argmax(c) for c in y])
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=422)
# del x,y

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

def train_xgb():
    clfs = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index], x[test_index]
        # trn_x, val_x = x_train[train_index], x_train[test_index]
        # trn_y, val_y = y_train[train_index], y_train[test_index]
        trn_y, val_y = y[train_index], y[test_index]
        clf = xgb.XGBRegressor(max_depth=10,
                               n_estimators=1500,
                               min_child_weight=9,
                               learning_rate=0.05,
                               nthread=8,
                               subsample=0.80,
                               colsample_bytree=0.80,
                               seed=4242)

        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=20)
        clfs.append(clf)
    return clfs

def make_submite():
    clfs = train_xgb()
    df = pd.read_csv('../stage2_sample_submission.csv')

    # import submit data 
    submit_data = np.load("../muchdata/muchdata_submit_stage2.npy",encoding='bytes')
    sub = []
    for j in submit_data:
        sub.append(j)
    # flatten submission data 
    sub = np.array([c.flatten() for c in sub])

    pid = [i.split("/")[-1].split("_")[0] for i in glob('../input_submit/*')]

    # predict 
    preds = []
    for clf in clfs:
        preds.append(np.clip(clf.predict(sub),0.001,1))
    pred = gmean(np.array(preds), axis=0)
    pred = (pred + pd.read_csv('stage2_submit1.csv').cancer) /2 
    submit_xgb = dict(zip(pid,pred))

    # ## predict accuracy
    # preds2 = []
    # for clf in clfs:
    #     preds2.append(np.clip(clf.predict(x_test),0.001,1))
    # pred2 = gmean(np.array(preds2), axis=0)
    # accuracy = [accuracy_score(y_test, pred2>c) for c in np.arange(0,1,0.01)]
    # test = pd.DataFrame(accuracy, index=np.arange(0,1,0.01), columns=['accuracy'])
    # test.plot()
    # plt.savefig('xgboost_3folder.png')


    # write data 
    df['cancer'] = df.id.map(submit_xgb)
    # df['cancer'] = df['cancer'].fillna(np.mean(df['cancer']))
    df.to_csv('stage2_submit2_ensmbled.csv', index=False)
    print(df.head())

if __name__ == '__main__':
    make_submite()