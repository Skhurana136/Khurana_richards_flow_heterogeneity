# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import pandas as pd
from DS.data_reader import data_processing as proc
from DS.analyses import transient as ta

# Unsaturated flow regime
raw_dir = r"D:\Data\Richards_flow\Richards_flow_tr"
output_dir = r"C:\Users\swkh9804\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
fpre = "RF-A"
fsuf = r"/"
gw = 0

scdict = proc.masterscenarios("Unsaturated") #master dictionary of all spatially heterogeneous scenarios that were run

# Default:
mTrial = list(t for t,values in scdict.items())
# Constants
yout = -6
yin = 6
xleft = 0
xright = -1
tr1 = 8 - gw
vely = 5
velx = 4
vars = [tr1]
vertnodes = 113
gvarnames = ["Tracer_study"]
vedge = 0.0025
velem = 0.005
vbc = 0.3
por = 0.2
Regimes = ["Medium", "Fast", "Slow"]
steps = [0.1,  0.01, 1.]
droplist = []

Trial = list(t for t in mTrial if t not in droplist)
total_inflow_list = []

for Reg in Regimes:
    d = os.path.join(raw_dir, Reg+"AR_0")
    filename = Reg+"AR_0_RF-AH_df.npy"
    data = np.load(os.path.join(d, filename))
    conctime, TotalFlow, Headinlettime, Total_inflow = ta.conc_time(data, yin, yout, xleft, xright, vertnodes, gvarnames, "Unsaturated", element_length = velem, edge_length = vedge)
    initial = Total_inflow
    for t in Trial:
        filename = Reg+"AR_0_RF-A"+t+"_df.npy"
        data = np.load(os.path.join(d, filename))
        conctime, TotalFlow, Headinlettime, Total_inflow = ta.conc_time(data, yin, yout, xleft, xright, vertnodes, gvarnames, "Unsaturated", element_length = velem, edge_length = vedge)
        total_inflow_list.append([Reg, t, scdict[t]['Het'], scdict[t]['Anis'], TotalFlow, Total_inflow/initial])

total_flow_data = pd.DataFrame(total_inflow_list, columns = ["Regime", "Trial", "Variance", "Anisotropy", "Total_inflow" ,"fraction"])
total_flow_f = os.path.join(output_dir, "total_inflow_woeffsat_03082022.csv")#11062021.csv")
total_flow_data.to_csv(total_flow_f, index = 'False')
