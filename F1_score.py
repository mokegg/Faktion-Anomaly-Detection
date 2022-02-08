#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:22:20 2022

@author: mokegg
"""

#Calculate the F1-score

import pandas as pd
import glob

path = 'Data/test_set/'
all_files = glob.glob(path + "/*.csv")
all_files
df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv( "merged.csv")
df_merged['prediction'].value_counts()
tp=df_merged['prediction'].value_counts()[1]
fp=df_merged['prediction'].value_counts()[0]
df_from_ano = pd.read_csv('Data/test_set/ano/csv_result_ano.csv')
df_from_ano['prediction'].value_counts()
fn = df_from_ano['prediction'].value_counts()[1]

F1 = tp/(tp+0.5*(fp+fn))
F1