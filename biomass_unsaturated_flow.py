# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:46:19 2021

@author: khurana
"""
import os
import numpy as np
import pandas as pd
import data_reader.data_processing as proc
import analyses.steady_state as ssa

# Unsaturated flow regime
Regimes = ["Medium", "Fast", "Slow"]
#fpre = "RF-A"
fsuf = r"/"
gw = 0
species = proc.speciesdict("Unsaturated")
gvarnames = list(t for t in species.keys() if (species[t]["State"]=="Active") or (species[t]["State"]=="Inactive"))

scdict = proc.masterscenarios() #master dictionary of all spatially heterogeneous scenarios that were run
horiznodes = 31
parent_dir = "X:/Richards_flow_big_sat"
# Default:
Trial = list(t for t,values in scdict.items())

# Constants
yout = -6
yin = 6
vertnodes = 63
xleft = 0
xright = -1
vedge = 0.005
velem = 0.01
vbc = 0.3
por = 0.2

row = []
for Reg in Regimes:
    for j in Trial:
        directory =  os.path.join(parent_dir, Reg + "AR_0")
        filename = Reg+"AR_0_RF-A"+str(j)+"_df.npy"
        data = np.load(os.path.join(directory, filename))
        sumbio = ssa.sum_biomass(data, yin, yout, xleft, xright, gvarnames, "Unsaturated")
        total = sum(sumbio)
        for g in gvarnames:
            print(Reg, j, g)
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, g, sumbio[gvarnames.index(g)], sumbio[gvarnames.index(g)]/total])

biomassdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "Biomass", "Biomass_contribution"])
biomassdata.Regime = biomassdata.Regime.replace({"Equal":"Medium"})
#Load tracer data
path_tr_data = "X:/Richards_flow/Tracer_studies/tracer_09012021.csv"
tr_data = pd.read_csv(path_tr_data, sep = "\t")
tr_data.columns

#Merge the datasets and save
cdata = pd.merge(biomassdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv("Y:/Home/khurana/4. Publications/Paper3/Figurecodes/biomass_10052021.csv", index=False)