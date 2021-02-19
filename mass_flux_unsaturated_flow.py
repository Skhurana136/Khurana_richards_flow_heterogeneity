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
import analyses.saturated_transient as sta

# Unsaturated flow regime
Regimes = ["Equal", "Fast", "Slow"]
#fpre = "RF-A"
fsuf = r"/"
gw = 0
species = proc.speciesdict("Unsaturated")
gvarnames = ["DOC","DO","Nitrate", "Ammonium","Nitrogen", "TOC"]

scdict = proc.masterscenarios() #master dictionary of all spatially heterogeneous scenarios that were run
horiznodes = 31
parent_dir = "E:/Richards_flow"
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
        directory =  os.path.join(parent_dir, Reg + "AR_0")
        filename = Reg+"AR_0_RF-A"+str(j)+"_df.npy"
        data = np.load(os.path.join(directory, filename))
        massfluxin, massfluxout = ssa.massflux(data, yin, yout, xleft, xright, gvarnames, "Unsaturated")
        delmassflux = massfluxin - massfluxout
        reldelmassflux = 100*delmassflux/massfluxin
        normmassflux = massfluxout/massfluxin
        for g in gvarnames:
            print(Reg, j, g)
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, g, massfluxin[gvarnames.index(g)], massfluxout[gvarnames.index(g)],delmassflux[gvarnames.index(g)], reldelmassflux[gvarnames.index(g)], normmassflux[gvarnames.index(g)]])

massfluxdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "massflux_in", "massflux_out","delmassflux", "reldelmassflux", "normmassflux"])
massfluxdata.Regime = massfluxdata.Regime.replace({"Equal":"Medium"})
#Load tracer data
path_tr_data = "X:/Richards_flow/Tracer_studies/tracer_09012021.csv"
tr_data = pd.read_csv(path_tr_data, sep = "\t")
tr_data.columns

#Merge the datasets and save
cdata = pd.merge(massfluxdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv("Y:/Home/khurana/4. Publications/Paper3/Figurecodes/massflux_10022021.csv", sep = "\t", index=False)

row = []
for Reg in Regimes:
    for j in Trial:
        directory =  os.path.join(parent_dir, Reg + "AR_0")
        filename = Reg+"AR_0_RF-A"+str(j)+"_df.npy"
        data = np.load(os.path.join(directory, filename))
        conctime, TotalFlow, Headinlettime = sta.conc_time(data, yin, yout, xleft, xright, 51, gvarnames, "Unsaturated")
        delconc = conctime[-1, 0, :] - conctime[-1,-1,:]
        reldelconc = 100*delconc/conctime[-1,0,:]
        normconc = conctime[-1,-1,:]/conctime[-1, 0, :]
        for g in gvarnames:
            print(Reg, j, g)
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, g, conctime[-1,0,gvarnames.index(g)], conctime[-1,-1,gvarnames.index(g)],delconc[gvarnames.index(g)], reldelconc[gvarnames.index(g)], normconc[gvarnames.index(g)]])

concdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "conc_in", "conc_out","delconc", "reldelconc", "normconc"])
concdata.Regime = massfluxdata.Regime.replace({"Equal":"Medium"})
#Load tracer data
path_tr_data = "X:/Richards_flow/Tracer_studies/tracer_09012021.csv"
tr_data = pd.read_csv(path_tr_data, sep = "\t")
tr_data.columns

#Merge the datasets and save
cdata = pd.merge(concdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv("Y:/Home/khurana/4. Publications/Paper3/Figurecodes/concdata_10022021.csv", sep = "\t", index=False)