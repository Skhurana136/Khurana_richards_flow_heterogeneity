# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:46:19 2021

@author: khurana
"""
import os
import numpy as np
import pandas as pd
import data_reader.data_processing as proc
import analyses.saturated_steady_state as sssa
import analyses.unsaturated_transient as uta

# Unsaturated flow regime
Regimes = ["Equal", "Fast", "Slow"]
#fpre = "RF-A"
fsuf = r"/"
gw = 0
species = proc.speciesdict("Unsaturated")
gvarnames = ["DOC","DO","Nitrate", "Ammonium","Nitrogen", "TOC"]

scdict = proc.masterscenarios() #master dictionary of all spatially heterogeneous scenarios that were run
horiznodes = 31

# Default:
Trial = list(t for t,values in scdict.items())

# Constants
yout = -1
yin = 0
xleft = 0
xright = -1
vedge = 0.005
velem = 0.01
vbc = 0.3
por = 0.2

row = []
for Reg in Regimes:
    for j in Trial:
        directory = "X:/Richards_flow/" + Reg + "AR_0/RF-A"+str(j)
        filename = "RF-A"+str(j)+".npy"
        data = np.load(os.path.join(directory, filename))
        massfluxin, massfluxout = sssa.calcmassfluxnew(data, yin, yout, xleft, xright, gvarnames, "Unsaturated")
        delmassflux = massfluxin - massfluxout
        reldelmassflux = 100*delmassflux/massfluxin
        normmassflux = massfluxout/massfluxin
        for g in gvarnames:
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, j, g, massfluxin[gvarnames.index(g)], massfluxout[gvarnames.index(g)],delmassflux[gvarnames.index(g)], reldelmassflux[gvarnames.index(g)], normmassflux[gvarnames.index(g)]])

massfluxdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Domain", "Regime", "Time_series", "Chem", "massflux_in", "massflux_out","delmassflux", "reldelmassflux", "normmassflux"])

#Load tracer data
path_tr_data = "X:/Richards_flow/Tracer_studies/tracer_09012021.csv"
tr_data = pd.read_csv(path_tr_data, sep = "\t")
tr_data.columns

#Merge the datasets and save
cdata = pd.merge(massfluxdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv("Y:/Home/khurana/4. Publications/Paper3/Figurecodes/massflux_10022021.csv", sep = "\t", index=False)
