import joblib
from scipy.stats import ttest_ind
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import pandas as pd 
import matplotlib.pyplot as plt 
import json
import os 
import numpy as np 


kmf_l = KaplanMeierFitter()
ESCA_kimiaNet_h = joblib.load('/path_to/test_all_folds_results.pkl')
ESCA_label = joblib.load('/path_to/esca_sur_and_time.pkl')
folds = joblib.load('/path_to/esca_five_folds.pkl')
hs = np.concatenate([ESCA_kimiaNet_h[k] for k in ESCA_kimiaNet_h.keys()])
mid = np.median(hs)
T_h = []
T_l = []
E_h = []
E_l = []
for k in ESCA_kimiaNet_h:

    if ESCA_kimiaNet_h[k]>=mid:
        T_l.append(ESCA_label[k][1])
        E_l.append(ESCA_label[k][0])
    else:
        T_h.append(ESCA_label[k][1])
        E_h.append(ESCA_label[k][0])

kmf_l.fit(T_l, event_observed=E_l)

kmf_h = KaplanMeierFitter()
kmf_h.fit(T_h, event_observed=E_h)

fig, ax = plt.subplots(figsize=(20, 12))
kmf_h.plot(ax=ax, show_censors=True,label='low risk')
kmf_l.plot(ax=ax, show_censors=True,label='high risk')


result = logrank_test(T_h,T_l,E_h,E_l)
print('p-value:',result.p_value)

ax.set_xlabel('Time (months)')
ax.set_ylabel('Cumulative Survival Probability')
ax.set_title('Kaplan-Meier Plot')
ax.figure.savefig(f'/path_to/ESCA-ours-p-value-{result.p_value}.png') # need rename picture name 