
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from .splitdf import split_df
from .info_value import iv
from .var_filter import var_filter
from .woebin import (woebin, woebin_ply)

with open('data/lr_model.json', 'r') as infile:
    attr = json.load(infile)

lr = LogisticRegression()
for k, v in attr.items():
    if isinstance(v, list):
        setattr(lr, k, np.array(v))
    else:
        setattr(lr, k, v)

# print(lr.__dict__)

with open('data/woe_bins.json', 'r') as infile:
    bins_adj = json.load(infile)

for k, v in bins_adj.items():
    bins_adj[k] = pd.read_json(v)

# print(bins_adj)
# print(lr.predict_proba(X))

dat = pd.read_csv('data/smalldataset.csv')

# print(dat)

# filter variable via missing rate, iv, identical value rate
# dt_s = var_filter(dat, y="creditability", iv_limit=0.07)

# print(dt_s)

# ivlist = iv(dt_s, y="creditability")
#
# print(ivlist)

# breaking dt into train and test
# train, test = split_df(dt_s, 'creditability', ratio=[0.7, 0.3]).values()
test = dat
print(test)

# # woe binning ------
#
# bins = woebin(dt_s, y="creditability")
# # woebin_plot(bins)
#
# # binning adjustment
# # # adjust breaks interactively
# # breaks_adj = woebin_adj(dt_s, "creditability", bins)
# # or specify breaks manually
# breaks_adj = {
#     'age.in.years': [27, 35, 45, 55],
#     'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]
# }
# bins_adj = woebin(dt_s, y="creditability", breaks_list=breaks_adj)

# converting train and test into woe values
# train_woe = woebin_ply(train, bins_adj)
test_woe = woebin_ply(test, bins_adj)

print(test_woe)

# y_train = train_woe.loc[:,'creditability']
# X_train = train_woe.loc[:,train_woe.columns != 'creditability']
y_test = test_woe.loc[:,'creditability']
X_test = test_woe.loc[:,test_woe.columns != 'creditability']

print("y_test {}, X_test {}".format(y_test, X_test))

# train_pred = lr.predict_proba(X_train)[:, 1]
test_pred = lr.predict_proba(X_test)[:, 1]

print(test_pred)
