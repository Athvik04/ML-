#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:32:09 2018

@author: raghav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_ann=pd.read_csv("ANN.csv")
x_ann=dataset_ann.iloc[:, 1].values
y_ann=dataset_ann.iloc[:, 2].values

dataset_fdm=pd.read_csv("FDM.csv")
x_fdm=dataset_fdm.iloc[:, 1].values
y_fdm=dataset_fdm.iloc[:, 2].values

plt.figure()
plt.plot(x_ann, y_ann, 'green')
plt.plot(x_fdm, y_fdm, 'blue')
plt.show()