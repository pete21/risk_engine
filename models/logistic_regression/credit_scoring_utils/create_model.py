
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .splitdf import split_df
from .info_value import iv
from .var_filter import var_filter
from .woebin import (woebin, woebin_ply)
from .perf import (perf_eva, perf_psi)
from .scorecard import (scorecard, scorecard_ply)


dat = pd.read_csv('data/creditdataset.csv')

# reorder columns alphabetically
# col_names = dat.columns.tolist()
# dat = dat[sorted(col_names)]

# filter variable via missing rate, iv, identical value rate
dt_s = var_filter(dat, y="creditability", iv_limit=0.07)

ivlist = iv(dt_s, y="creditability")

# breaking dt into train and test
train, test = split_df(dt_s, 'creditability', ratio=[0.7, 0.3]).values()

# woe binning ------

bins = woebin(dt_s, y="creditability")
# woebin_plot(bins)

# binning adjustment
# # adjust breaks interactively
# breaks_adj = woebin_adj(dt_s, "creditability", bins)
# or specify breaks manually
breaks_adj = {
    'age.in.years': [27, 35, 45, 55],
    'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]
}
bins_adj = woebin(dt_s, y="creditability", breaks_list=breaks_adj)

# converting train and test into woe values
train_woe = woebin_ply(train, bins_adj)
test_woe = woebin_ply(test, bins_adj)

y_train = train_woe.loc[:,'creditability']
X_train = train_woe.loc[:,train_woe.columns != 'creditability']
y_test = test_woe.loc[:,'creditability']
X_test = test_woe.loc[:,train_woe.columns != 'creditability']

print(X_train)

# logistic regression ------

lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
print(lr.coef_)
print("------------------------")
print(lr.intercept_)
print("------------------------")
attr = lr.__dict__
print(attr)

for k, v in attr.items():
    if isinstance(v, np.ndarray):
        attr[k] = v.tolist()

with open('data/lr_model.json', 'w') as outfile:
    json.dump(attr, outfile)

for k, v in bins_adj.items():
    bins_adj[k] = v.to_json()

with open('data/woe_bins.json', 'w') as outfile:
    json.dump(bins_adj, outfile)
