# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:46:19 2021

@author: khurana
"""
import os
import numpy as np
import pandas as pd
import DS.data_reader.data_processing as proc
import DS.analyses.steady_state as ssa

# Unsaturated flow regime
Regimes = ["Medium", "Fast", "Slow"]
#fpre = "RF-A"
fsuf = r"/"
gw = 0
species = proc.speciesdict("Unsaturated")
gvarnames = list(t for t in species.keys() if (species[t]["State"]=="Active") or (species[t]["State"]=="Inactive"))

scdict = proc.masterscenarios("Unsaturated") #master dictionary of all spatially heterogeneous scenarios that were run
horiznodes = 61
parent_dir = r"D:\Data\Richards_flow\RF_big_sat_2"
results_dir = r"C:\Users\swkh9804\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
# Default:
droplist = []
Trial = list(t for t,values in scdict.items() if t not in droplist)

# Constants
yout = -7
yin = 6
vertnodes = 113
xleft = 0
xright = -1
vedge = 0.0025
velem = 0.005
vbc = 0.3
por = 0.2

row = []
for Reg in Regimes:
    for j in Trial:
        directory = parent_dir
        filename = Reg+"AR_0_RF-A"+str(j)+"_df.npy"
        data = np.load(os.path.join(directory, filename))
        sat = np.mean(data[4,-1,yin:yout+1,:])
        sumbio = ssa.sum_biomass(data, yin, yout, xleft, xright, gvarnames, "Unsaturated")
        total = sum(sumbio)
        for g in gvarnames:
            print(Reg, j, g)
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, g, sumbio[gvarnames.index(g)], sumbio[gvarnames.index(g)]/total, sat])

biomassdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "Biomass", "Biomass_contribution", "Mean_saturation"])
biomassdata.Regime = biomassdata.Regime.replace({"Equal":"Medium"})
#Load tracer data
path_tr_data = os.path.join(results_dir,"tracer_09082022.csv")
tr_data = pd.read_csv(path_tr_data)
tr_data.columns

#Merge the datasets and save
cdata = pd.merge(biomassdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv(os.path.join(results_dir,"biomass_09082022.csv"), index=False)

row = []
for Reg in Regimes:
    for j in Trial:
        directory =  parent_dir
        filename = Reg+"AR_0_RF-A"+str(j)+"_df.npy"
        data = np.load(os.path.join(directory, filename))
        sat = np.mean(data[4,-1,yin:yout+1,:])
        sumbio = ssa.sum_biomass(data, yin, yout, xleft, xright, gvarnames, "Unsaturated")
        total = sum(sumbio)
        for g in gvarnames:
            print(Reg, j, g)
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, g, sumbio[gvarnames.index(g)], sumbio[gvarnames.index(g)]/total, sat])

biomassdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "Biomass", "Biomass_contribution", "Mean_saturation"])
biomassdata.Regime = biomassdata.Regime.replace({"Equal":"Medium"})
#Load tracer data
path_tr_data = os.path.join(results_dir,"tracer_09082022.csv")
tr_data = pd.read_csv(path_tr_data)
tr_data.columns

#Merge the datasets and save
cdata = pd.merge(biomassdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv(os.path.join(results_dir,"biomass_09082022.csv"), index=False)
