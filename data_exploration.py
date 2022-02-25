# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:46:19 2021

@author: khurana
"""
## Libraries
# %%
# Import python libraries and data exploration packages
import os
import numpy as np
import pandas as pd
#Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#User libraries
from DS.data_reader import data_processing as proc
print ("All libraries loaded")

#%%
# Assign directories based on where you are
# work computer
data_dir = r"C:\Users\swkh9804\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
raw_dir = r"D:\Data\Richards_flow\RF_big_sat_2"
# %%
#Assign file names
biomass_file = "biomass_comparison_with_sat_26092021.csv"
chem_file = "conc_comparison_with_sat_26092021.csv"
print ("Path directories and files assigned")

## Load datasets and explore
#%%
# Exploring biomass data set
biomass_path_data = os.path.join(data_dir, biomass_file)
biomassdata = pd.read_csv(biomass_path_data)
print ("Biomass data loaded and read")
print(biomassdata.shape)
print(biomassdata.columns)

#%%
# Exploring chemical data set
chem_path_data = os.path.join(data_dir, chem_file)
chemdata = pd.read_csv(chem_path_data)
print ("Chemical flux data loaded and read")
print(chemdata.columns)
chemdata['Regime'] = chemdata['Regime'].replace({'Equal':'Medium'})
print(chemdata.Regime.unique())

## Scatter plots of biomass data with each chemical
#%%
# Identify unique variables in both datasets
biomass_vars = biomassdata.Chem.unique().tolist()
biomass_vars = list(b for b in biomass_vars if "sulphate" not in b)
print(biomass_vars)
#chem_vars = chemdata.Chem.unique().tolist().remove("Nitrogen").remove("TOC")
chem_vars = ["DOC", "DO", "Nitrate", "Ammonium"]
print(chem_vars)
regimes = chemdata.Regime.unique().tolist()
color_scheme = {"Slow":"darkorange",
"Medium":"limegreen", "Fast":"slateblue"}
marker_scheme = {"Slow":"o", "Medium":"s", "Fast":"^"}

#%%
# Make a grid plot with each biomass fraction against each chemical
for state in ["Immobile", "Mobile"]:
    biomass_vars_plot = list (b for b in biomass_vars if state in b)
    fig, ax = plt.subplots(len(biomass_vars_plot),len(chem_vars),figsize = (20,20),
    sharex = "col", sharey = "row")
    for b in biomass_vars_plot:
        bio_sub = biomassdata[biomassdata.Chem==b]
        bio_row = biomass_vars_plot.index(b)
        for c in chem_vars:
            chem_sub = chemdata[chemdata.Chem==c]
            chem_row = chem_vars.index(c)
            ax_index = bio_row*len(chem_vars) + chem_row
            subax = ax.flat[ax_index]
            for reg in regimes:
                bio_reg = bio_sub[bio_sub.Regime == reg]
                chem_reg = chem_sub[chem_sub.Regime == reg]
                subax.scatter(chem_reg.conc_out, bio_reg.Biomass,
                color = color_scheme[reg], marker = marker_scheme[reg])
            if bio_row == 0:
                subax.set_title(c)
            if chem_row == 0:
                subax.set_ylabel(b)
            else:
                subax.set_ylabel("")
            subax.set_xscale("log")
    picname = os.path.join(data_dir, "scatter"+state+".pdf")
    fig.savefig(picname, dpi = 300, pad = 0.1)

#%%
# Identify oxic domains
import DS.analyses.transient as sta
Regimes = ["Slow", "Medium", "Fast"]
scdict = proc.masterscenarios("Unsaturated")
species = proc.speciesdict("Unsaturated")
yout = -6
yin = 6
vertnodes = 113
xleft = 0
xright = -1
vedge = 0.0025
velem = 0.005
vbc = 0.3
por = 0.2

dom = []
for r in Regimes:
    for j in list(scdict.keys()):
        file = os.path.join(raw_dir, r+"AR_0_RF-A"+str(j)+"_df.npy")
        data = np.load(file)
        sat = np.mean(data[4,-1,6:-6,:])
        conctime, TotalFlow, Headinlettime = sta.conc_time(data, yin, yout, xleft, xright, vertnodes, ["DO"], "Unsaturated")
        do_data = conctime[-1,yin:yout,0]
        dom.append([r, j,scdict[j]['Het'], scdict[j]['Anis'], np.mean(do_data), np.median(do_data), do_data[-1], sat])

#%%
dom_df = pd.DataFrame.from_records(dom, columns = ["Regime","Trial","Variance","Anisotropy","Mean_DO","Median_DO","Outlet_DO", "Saturation"])
dom_df.to_csv(os.path.join(data_dir, "oxic_state.csv"))

#%%
anox_dom = dom_df[dom_df.Median_DO < 50]
print(anox_dom.shape)
print(anox_dom.Regime.unique())
ox_dom = dom_df.loc[dom_df.index.difference(anox_dom.index), ]
print(ox_dom.shape)
print(ox_dom.Regime.unique())

#%%
plt.figure()
sns.scatterplot(x = "Saturation", y = "Mean_DO", hue = "Regime", data = anox_dom, marker = '+', hue_order = ["Slow","Medium","Fast"])
sns.scatterplot(x = "Saturation", y = "Mean_DO", hue = "Regime", data = ox_dom, marker = '^', hue_order = ["Slow","Medium","Fast"])
#%%
anox_scenarios = anox_dom.index.tolist()
#%%
sns.scatterplot(x="Variance", y="Mean_DO", data = anox_dom, hue = "Regime", marker = "o")
sns.scatterplot(x="Variance", y="Median_DO", data = anox_dom, hue = "Regime", marker = "^")
#sns.scatterplot(x="Variance", y="Mean_DO", data = anox_dom)
#%%
sns.scatterplot(x="Variance", y="Mean_DO", data = ox_dom, hue = "Regime", marker = "o")
#sns.scatterplot(x="Variance", y="Median_DO", data = ox_dom, hue = "Regime", marker = "^")
plt.legend(loc = (0, -0.5), ncol = 3)