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
import DS.analyses.transient as sta

# Unsaturated flow regime
Regimes = ["Medium", "Fast", "Slow"]
#fpre = "RF-A"
fsuf = r"/"
gw = 0
species = proc.speciesdict("Unsaturated")
gvarnames = ["DO"]

scdict = proc.masterscenarios("Unsaturated") #master dictionary of all spatially heterogeneous scenarios that were run
parent_dir = r"D:\Data\Richards_flow\RF_big_sat_2"
results_dir = r"C:\Users\swkh9804\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
# Default:
droplist = []#["38","48","72","82","114","116"]
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
        directory =  os.path.join(parent_dir)#, Reg + "AR_0")
        filename = Reg+"AR_0_RF-A"+str(j)+"_df.npy"
        data = np.load(os.path.join(directory, filename))
        conctime, TotalFlow, Headinlettime = sta.conc_time(data, yin, yout, xleft, xright, vertnodes, gvarnames, "Unsaturated")
        condition_oxic_check = np.where(conctime[-1,yin:yout+1]>20)
        pc_oxic = 100*np.shape(condition_oxic_check)[1]/101
        for g in gvarnames:
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], Reg, g, pc_oxic])

massfluxdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "percentage_oxic"])
massfluxdata.Regime = massfluxdata.Regime.replace({"Equal":"Medium"})
massfluxdata.to_csv(os.path.join(results_dir,"pc_oxic_09082022.csv"), index=False)

