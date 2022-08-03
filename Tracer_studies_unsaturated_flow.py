# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import pandas as pd
from DS.data_reader import data_processing as proc
from DS.analyses import transient as ta
from DS.plots import general_plots as gp
#import analyses.transient as ta
#import plots.general_plots as gp

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
Regimes = ["Slow", "Medium", "Fast"]
steps = [2., 0.1,  0.01]
droplist = []

Trial = list(t for t in mTrial if t not in droplist)

breakthrough2 = []
for Reg, s in zip(Regimes, steps):
    d = os.path.join(raw_dir, Reg+"AR_0")
    filename = Reg+"AR_0_RF-AH_df.npy"
    data = np.load(os.path.join(d, filename))
    conctime, TotalFlow, Headinlettime = ta.conc_time(data, 0, -1, xleft, xright, vertnodes, gvarnames, "Unsaturated", element_length = velem, edge_length = vedge)
    Time =  np.where(np.round(conctime[:, yout, 0], 3) > 10)
    initial = s*Time[0][0]
    for t in Trial:
        filename = Reg+"AR_0_RF-A"+t+"_df.npy"
        data = np.load(os.path.join(d, filename))
        conctime, TotalFlow, Headinlettime = ta.conc_time(data, 0, -1, xleft, xright, vertnodes, gvarnames, "Unsaturated", element_length = velem, edge_length = vedge)
        Time = np.where(np.round(conctime[:, yout, 0], 3) > 10)
        print(s * Time[0][0], initial, (s * Time[0][0]) / initial)
        breakthrough2.append([Reg, t, scdict[t]['Het'], scdict[t]['Anis'], "Tracer", s*Time[0][0], (s*Time[0][0])/initial])

data = pd.DataFrame(breakthrough2, columns = ["Regime", "Trial", "Variance", "Anisotropy", "Chem", "Time", "fraction"])
f = os.path.join(output_dir, "tracer_wo_effsat_03082022.csv")#11062021.csv")
data.to_csv(f, index = 'False')
        
## plotting boxplots to see variance of breakthrough from homogeneous scenario
tracerplot = gp.plot_tracer(f)
plot_path = os.path.join(output_dir, "tracer_breakthrough_impact_wo_effsat_03082022.png")#"tracer_breakthrough_impact.png")
tracerplot.savefig(plot_path, dpi = 300, pad_inches = 0.1, bbox_inches = 'tight')
