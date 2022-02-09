#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:22:20 2022

@author: mokegg
"""

#Calculate the F1-score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_n = pd.read_csv('Data/templates/csv_all_test.csv')

df_a = pd.read_csv('Data/templates/csv_all_test_ano.csv')

ab_no = []
for i in np.arange(0.60,0.87,0.01):
    i = round(i, 3)
    fp = df_a[df_a.msssim >= i].shape[0]
    fn = df_n[df_n.msssim <= i].shape[0]
    sum= fp + fn
    tp = df_n.shape[0]-fn
    F1 = round(tp/(tp+0.5*(fp+fn)),3)
    
    ab_no.append([i, fn, fp, sum, F1])
df_f1 = pd.DataFrame(ab_no, columns = ['msssim', 'fn', 'fp', 'sum', 'F1_score'])    

F1_opt = df_f1.iloc[df_f1['F1_score'].idxmax()].F1_score
opt_msssim = df_f1.iloc[df_f1['F1_score'].idxmax() | df_f1['fn'].idxmin()].msssim
 
print(df_f1)    
print(f'optimum MSSSIM (multi-scale structural similarity index) = {opt_msssim} ---> F1 score = { F1_opt}')
# Visualazation

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(df_f1.msssim, df_f1.F1_score, color="red", marker="o")
# set x-axis label
ax.set_xlabel("MS_SSIM",fontsize=14)
# set y-axis label
ax.set_ylabel("F1 Score",color="red",fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(df_f1.msssim, df_f1.fp,color="blue",marker="o", label= "False Positives")
ax2.set_ylabel("False predictions",color="blue",fontsize=14)
ax2.plot(df_f1.msssim, df_f1.fn,color="green",marker="o", label= "False Negatives")
ax2.legend(loc='upper left', bbox_to_anchor= (1.1, 0.6), ncol=1,
            borderaxespad=0, frameon=False)


plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')

