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
print ("All libraries loaded")

# %%
#Assign file names and assign directories
data_dir = "D:\Publications\Paper3\Figurecodes"
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
chem_vars = chemdata.Chem.unique().tolist()
print(chem_vars)
regimes = chemdata.Regime.unique().tolist()
color_scheme = {"Slow":"darkorange",
"Medium":"limegreen", "Fast":"slateblue"}
marker_scheme = {"Slow":"o",
"Medium":"s", "Fast":"^"}

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