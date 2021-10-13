# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import pandas as pd
import data_reader.data_processing as proc
import analyses.transient as ta
import plots.general_plots as gp

# Unsaturated flow regime
raw_dir = "E:/Richards_flow/Richards_flow_tr"
output_dir = "Y:/Home/khurana/4. Publications/Paper3/Figurecodes"
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

breakthrough2 = []
for Reg, s in zip(Regimes, steps):
    d = os.path.join(raw_dir, Reg+"AR_0")
    filename = Reg+"AR_0_RF-AH_df.npy"
    data = np.load(os.path.join(d, filename))
    conctime, TotalFlow, Headinlettime = ta.conc_time(data, yin, yout, xleft, xright, vertnodes, gvarnames, "Unsaturated")
    Time = np.where(np.round(conctime[:, yout, 0], 3) > 10)
    initial = s*Time[0][0]
    for t in Trial:
        filename = Reg+"AR_0_RF-A"+t+"_df.npy"
        data = np.load(os.path.join(d, filename))
        conctime, TotalFlow, Headinlettime = ta.conc_time(data, yin, yout, xleft, xright, vertnodes, gvarnames, "Unsaturated")
        Time = np.where(np.round(conctime[:, yout, 0], 3) > 10)
        print(s * Time[0][0], initial, (s * Time[0][0]) / initial)
        breakthrough2.append([Reg, t, scdict[t]['Het'], scdict[t]['Anis'], "Tracer", s*Time[0][0], (s*Time[0][0])/initial])

data = pd.DataFrame(breakthrough2, columns = ["Regime", "Trial", "Variance", "Anisotropy", "Chem", "Time", "fraction"])
f = os.path.join(output_dir, "tracer_11062021.csv")
data.to_csv(f, index = 'False')
        
# plotting boxplots to see variance of breakthrough from homogeneous scenario
tracerplot = gp.plot_tracer(f)
plot_path = os.path.join(output_dir, "tracer_breakthrough_impact.png")
tracerplot.savefig(plot_path, dpi = 300, pad_inches = 0.1, bbox_inches = 'tight')
