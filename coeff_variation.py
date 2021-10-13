# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:46:19 2021

@author: khurana
"""
import os
import numpy as np
import pandas as pd
import data_reader.data_processing as proc

# Unsaturated flow regime
Regimes = ["Medium", "Fast", "Slow"]
fsuf = r"/"
gw = 0
species = proc.speciesdict("Unsaturated")
species_num = len(list(species.keys()))

scdict = proc.masterscenarios("Unsaturated") #master dictionary of all spatially heterogeneous scenarios that were run
parent_dir = "E:/Richards_flow_/RF_big_sat_2"
results_dir = "Y:/Home/khurana/4. Publications/Restructuring/Thesis/Figures"
# Default:
droplist = []
Trial = list(t for t,values in scdict.items() if t not in droplist)

# Constants
yout = -6
yin = 6
vertnodes = 113
xleft = 0
xright = -1
vedge = 0.0025
velem = 0.005
vbc = 0.3
por = 0.2

row = []
for reg in Regimes:
    for t in Trial:
        print(reg,t)
        file = os.path.join(parent_dir, reg + "AR_0", reg + "AR_0_RF-A"+str(t)+"_df.npy")
        data = np.load(file)
        shaped_data = data[:,-1,yin:yout,:].reshape(28,6161)
        #scaled_data = shaped_data#scale(shaped_data)
        #mean_array = np.zeros((species_num,1))
        #std_array = np.zeros((species_num,1))
        for g,i in zip(list(species.keys()), list(range(species_num))):
            #cv_array[i] = np.apply_along_axis(cv, axis = 0, arr = scaled_data[species[g]['TecIndex'],:])
            m = np.mean(shaped_data[species[g]['TecIndex'],:])
            std = np.std(shaped_data[species[g]['TecIndex'],:])
            row.append([reg, t, scdict[t]['Het'], scdict[t]['Anis'],g, m, std])
        sat_m = np.mean(shaped_data[4,:])#cv(scaled_data[4,:])
        sat_std = np.std(shaped_data[4,:])
        vel_y_m = np.mean(shaped_data[2,:])#cv(scaled_data[2,:])
        vel_y_std = np.std(shaped_data[2,:])
        row.append([reg, t, scdict[t]['Het'], scdict[t]['Anis'],"Saturation", sat_m, sat_std])
        row.append([reg, t, scdict[t]['Het'], scdict[t]['Anis'],"vel_y", vel_y_m, vel_y_std])

cvdata = pd.DataFrame.from_records (row, columns = ["Regime", "Trial", "Variance", "Anisotropy", "Chem", "Mean", "Sdev"])
#Load tracer data
path_tr_data = os.path.join(results_dir,"tracer_11062021.csv")
tr_data = pd.read_csv(path_tr_data)
tr_data.columns

#Merge the datasets and save
cvdata = pd.merge(cvdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cvdata['cv'] = cvdata.Sdev/cvdata.Mean

cvdata.to_csv(os.path.join(results_dir,"coeff_var.csv"), index=False)

cvdata.cv.describe()

cvdata = pd.read_csv(os.path.join(results_dir,"coeff_var.csv"))
datah = cvdata[(cvdata.Trial=='H')&(cvdata.Chem=='vel_y')]
datah[['Regime','Chem','Time']]
