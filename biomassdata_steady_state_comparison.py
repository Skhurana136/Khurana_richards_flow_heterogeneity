# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:05:06 2020

@author: khurana
"""

import pandas as pd

#Load data
path_data = "Y:/Home/khurana/4. Publications/Paper3/Figurecodes/biomass_16022021.csv"
data = pd.read_csv(path_data, sep = "\t")
data.columns
data.dtypes

regimes = data.Regime.unique().tolist()
chem_series = data.Chem.unique().tolist()
trial_series = data.Trial.unique().tolist()
spatial_base = 'H'
temp_base = 0

for r in regimes:
    for t in trial_series:
        for c in chem_series:
            #spat_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['normmassflux'].values[0]
            spat_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['Biomass'].values[0]
            #tim_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['normmassflux'].values[0]
            #spat_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['reldelmassflux'].values[0]
            spat_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['Biomass_contribution'].values[0]
            #tim_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['reldelmassflux'].values[0]
            #data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'temporal_normmassflux_base'] = tim_n_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_biomass_base'] = spat_n_base
            #data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'temporal_reldelmassflux_base'] = tim_r_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_biomass_contribution_base'] = spat_r_base
        
#data['normmassflux_temporal_fraction'] = data['normmassflux']/data['temporal_normmassflux_base']
data['biomass_spatial_fraction'] = data['Biomass']/data['spatial_biomass_base']
#data['reldelmassflux_temporal_fraction'] = data['reldelmassflux']/data['temporal_reldelmassflux_base']
data['biomass_contribution_spatial_fraction'] = data['Biomass_contribution']/data['spatial_biomass_contribution_base']

data.to_csv("Y:/Home/khurana/4. Publications/Paper3/Figurecodes/biomass_comparison_16022021.csv", sep ="\t", index = False)
