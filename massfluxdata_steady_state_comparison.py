# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:05:06 2020

@author: khurana
"""
import os
import pandas as pd
import numpy as np

results_dir = "Y:/Home/khurana/4. Publications/Paper3/Figurecodes"
#Load data
path_data = os.path.join(results_dir, "massflux_with_sat_26092021.csv")
data = pd.read_csv(path_data)
data.columns
data.dtypes

regimes = data.Regime.unique().tolist()
#Time_series = data.Time_series.unique().tolist()
chem_series = data.Chem.unique().tolist()
trial_series = data.Trial.unique().tolist()
spatial_base = 'H'
temp_base = 0

for r in regimes:
    for t in trial_series:
        for c in chem_series:
            #spat_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['normmassflux'].values[0]
            spat_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['normmassflux'].values[0]
            #tim_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['normmassflux'].values[0]
            #spat_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['reldelmassflux'].values[0]
            spat_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['reldelmassflux'].values[0]
            #tim_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['reldelmassflux'].values[0]
            #data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'temporal_normmassflux_base'] = tim_n_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_normmassflux_base'] = spat_n_base
            #data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'temporal_reldelmassflux_base'] = tim_r_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_reldelmassflux_base'] = spat_r_base
        
#data['normmassflux_temporal_fraction'] = data['normmassflux']/data['temporal_normmassflux_base']
data['normmassflux_spatial_fraction'] = data['normmassflux']/data['spatial_normmassflux_base']
#data['reldelmassflux_temporal_fraction'] = data['reldelmassflux']/data['temporal_reldelmassflux_base']
data['reldelmassflux_spatial_fraction'] = data['reldelmassflux']/data['spatial_reldelmassflux_base']
data["Da63"] = np.log10(data.normmassflux)

data.to_csv(os.path.join(results_dir,"massflux_comparison_with_sat_26092021.csv"),
            index = False)

#Load data
path_data = os.path.join(results_dir,"concdata_with_sat_26092021.csv")
data = pd.read_csv(path_data)
data.columns
data.dtypes

regimes = data.Regime.unique().tolist()
#Time_series = data.Time_series.unique().tolist()
chem_series = data.Chem.unique().tolist()
trial_series = data.Trial.unique().tolist()
spatial_base = 'H'
temp_base = 0

for r in regimes:
    for t in trial_series:
        for c in chem_series:
            #spat_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['normmassflux'].values[0]
            spat_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['normconc'].values[0]
            #tim_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['normmassflux'].values[0]
            #spat_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['reldelmassflux'].values[0]
            spat_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['reldelconc'].values[0]
            #tim_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['reldelmassflux'].values[0]
            #data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'temporal_normmassflux_base'] = tim_n_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_normconc_base'] = spat_n_base
            #data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'temporal_reldelmassflux_base'] = tim_r_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_reldelconc_base'] = spat_r_base
        
#data['normmassflux_temporal_fraction'] = data['normmassflux']/data['temporal_normmassflux_base']
data['normconc_spatial_fraction'] = data['normconc']/data['spatial_normconc_base']
#data['reldelmassflux_temporal_fraction'] = data['reldelmassflux']/data['temporal_reldelmassflux_base']
data['reldelconc_spatial_fraction'] = data['reldelconc']/data['spatial_reldelconc_base']
data["Da63"] = np.log10(data.normconc)

data.to_csv(os.path.join(results_dir,"conc_comparison_with_sat_26092021.csv"), index = False)
