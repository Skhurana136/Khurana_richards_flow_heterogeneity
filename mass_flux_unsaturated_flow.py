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
import analyses.transient as sta

# Unsaturated flow regime
Regimes = ["Medium", "Fast", "Slow"]
#fpre = "RF-A"
fsuf = r"/"
gw = 0
species = proc.speciesdict("Unsaturated")
gvarnames = ["DOC","DO","Nitrate", "Ammonium","Nitrogen", "TOC"]

scdict = proc.masterscenarios("Unsaturated") #master dictionary of all spatially heterogeneous scenarios that were run
parent_dir = "X:/Richards_flow_big_sat"
results_dir = "Y:/Home/khurana/4. Publications/Paper3/Figurecodes"
# Default:
droplist = []#["38","48","72","82","114","116"]
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
for Reg in Regimes:
    for j in Trial:
        directory =  os.path.join(parent_dir, Reg + "AR_0")
        filename = Reg+"AR_0_RF-A"+str(j)+"_df.npy"
        data = np.load(os.path.join(directory, filename))
        sat = np.mean(data[4,-1,6:-6,:])
        massfluxin, massfluxout = ssa.massflux(data, yin, yout, xleft, xright, gvarnames, "Unsaturated")
        delmassflux = massfluxin - massfluxout
        reldelmassflux = 100*delmassflux/massfluxin
        normmassflux = massfluxout/massfluxin
        for g in gvarnames:
            #print(Reg, j, g)
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, g, massfluxin[gvarnames.index(g)], massfluxout[gvarnames.index(g)],delmassflux[gvarnames.index(g)], reldelmassflux[gvarnames.index(g)], normmassflux[gvarnames.index(g)],sat])

massfluxdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "massflux_in", "massflux_out","delmassflux", "reldelmassflux", "normmassflux", "Mean_saturation"])
massfluxdata.Regime = massfluxdata.Regime.replace({"Equal":"Medium"})
#Load tracer data
path_tr_data = os.path.join(results_dir,"tracer_11062021.csv")
tr_data = pd.read_csv(path_tr_data)
tr_data.columns

#Merge the datasets and save
cdata = pd.merge(massfluxdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv(os.path.join(results_dir,"massflux_with_sat_26092021.csv"), index=False)

row = []
for Reg in Regimes:
    for j in Trial:
        directory =  os.path.join(parent_dir, Reg + "AR_0")
        filename = Reg+"AR_0_RF-A"+str(j)+"_df.npy"
        data = np.load(os.path.join(directory, filename))
        sat = np.mean(data[4,-1,6:-6,:])
        conctime, TotalFlow, Headinlettime = sta.conc_time(data, yin, yout, xleft, xright, vertnodes, gvarnames, "Unsaturated")
        delconc = conctime[-1, yin, :] - conctime[-1,yout,:]
        reldelconc = 100*delconc/conctime[-1,yin,:]
        normconc = conctime[-1,yout,:]/conctime[-1, yin, :]
        for g in gvarnames:
            print(Reg, j, g)
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, g, conctime[-1,yin,gvarnames.index(g)], conctime[-1,yout,gvarnames.index(g)],delconc[gvarnames.index(g)], reldelconc[gvarnames.index(g)], normconc[gvarnames.index(g)], sat])

concdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "conc_in", "conc_out","delconc", "reldelconc", "normconc", "Mean_saturation"])
concdata.Regime = massfluxdata.Regime.replace({"Equal":"Medium"})

#Merge the datasets and save
cdata = pd.merge(concdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv(os.path.join(results_dir,"concdata_with_sat_26092021.csv"), index=False)