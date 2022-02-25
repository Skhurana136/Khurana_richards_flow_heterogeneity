# -*- coding: utf-8 -*-
"""
This is a script file to generate graphics for a publication
to be submitted to Vadose Zone Journal.
"""
#%%
#Native libraries
import os

#Third party processing libraries
import pandas as pd
import numpy as np

#Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl  
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

#User libraries
from DS.data_reader import data_processing as proc
from DS.data_reader.data_processing import tracerstudies
import DS.analyses.transient as sta

#uss_raw_dir = "E:/Richards_flow/RF_big_sat_2"
uss_dir = r"C:\Users\swami\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
op_dir = r"C:\Users\swami\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"

#%%
#Standard color and font options
my_pal = {2:"indianred", 11:"g", 22:"steelblue", "DO":"indianred", "Nitrate":"g", "Ammonium":"steelblue",
          'Slow':"indianred", "Medium":"g", "Fast":"steelblue"}
legendkw = {'fontsize' : 14}
labelkw = {'labelsize' : 14}
secondlabelkw = {'labelsize' : 16}
suptitlekw = {'fontsize' : 18}
titlekw = {'fontsize' : 16}
mpl.rc('font',family='Arial')
legendsize = 16
axissize = 16
ticksize = 14
titlesize = 20

Regimes = ["Slow", "Medium", "Fast"]
trialist = proc.masterscenarios("Unsaturated")
species = proc.speciesdict("Unsaturated")
gvarnames = ["DO", "DOC", "Ammonium", "Nitrate"]

unbio_file = os.path.join(uss_dir,"biomass_comparison_with_sat_26092021.csv")#biomass_comparison_03082021.csv")
unbio_data = pd.read_csv(unbio_file)
#Plotting
species = ["Immobile active aerobic degraders", "Immobile active ammonia oxidizers","Immobile active nitrate reducers"]
labels = ["Aerobic degraders", "Ammonia oxidizers","Nitrate reducers"]
#Reactive species of concerns
States = ["Active", "Inactive"]
Locations = ["Mobile", "Immobile"]
allspecies = proc.speciesdict("Unsaturated")
microbialspecies = list(t for t in allspecies.keys() if allspecies[t]["State"] in States)
print(unbio_data.columns)
unbio_data['Regime'] = unbio_data['Regime'].replace({'Equal':'Medium'})
print(unbio_data.Regime.unique())
print(unbio_data.shape)

for s in microbialspecies:
    unbio_data.loc[unbio_data.Chem == s, 'State'] = allspecies[s]["State"]
    unbio_data.loc[unbio_data.Chem == s, 'Location'] = allspecies[s]['Location']

state_df = pd.DataFrame(columns = ["Regime", "Trial", "Mass_Active", "Mass_Inactive", "State_Ratio","Variance","Anisotropy","Time","fraction","Sat"])
loc_df = pd.DataFrame(columns = ["Regime", "Trial", "Mass_Immobile", "Mass_Mobile", "Loc_Ratio","Variance","Anisotropy","Time","fraction","Sat"])
column = "Biomass"
x_state = unbio_data.groupby(['Regime','Trial','State'], as_index=False)[column].sum().pivot_table(index = ['Regime','Trial'], columns = ['State'], values = [column]).reset_index()
x_loc = unbio_data.groupby(['Regime','Trial','Location'], as_index=False)[column].sum().pivot_table(index = ['Regime','Trial'], columns = ['Location'], values = [column]).reset_index()
x_state.columns = ["_".join(a) for a in x_state.columns.to_flat_index()]
x_loc.columns = ["_".join(a) for a in x_loc.columns.to_flat_index()]
x_state['State_Ratio'] = x_state[column+'_Active']/x_state[column+'_Inactive']
x_loc['Loc_Ratio'] = x_loc[column+'_Immobile']/x_loc[column+'_Mobile']
x_state.rename(columns={'Regime_':'Regime'}, inplace=True)
x_loc.rename(columns={'Regime_':'Regime'}, inplace=True)
x_state.rename(columns={'Trial_':'Trial'}, inplace=True)
x_loc.rename(columns={'Trial_':'Trial'}, inplace=True)
shortbiomass = unbio_data[unbio_data['Chem']==microbialspecies[0]]
x_state = x_state.merge(shortbiomass[['Regime','Trial','Variance','Anisotropy','Time','fraction','Mean_saturation']], how='left', on=['Regime','Trial'])
x_loc = x_loc.merge(shortbiomass[['Regime','Trial','Variance','Anisotropy','Time','fraction','Mean_saturation']], how='left', on=['Regime','Trial'])
x_state.rename(columns={'Mean_saturation':'Sat'}, inplace=True)
x_loc.rename(columns={'Mean_saturation':'Sat'}, inplace=True)
x_state.rename(columns={'Biomass_Active':'Mass_Active', 'Biomass_Inactive':'Mass_Inactive'}, inplace=True)
x_loc.rename(columns={'Biomass_Immobile':'Mass_Immobile', 'Biomass_Mobile':'Mass_Mobile'}, inplace=True)
state_df = pd.concat([state_df, x_state], axis = 0, ignore_index = True)
loc_df = pd.concat([loc_df, x_loc], axis = 0, ignore_index = True)
    
biomass_ratio_df = state_df.merge(loc_df[['Regime','Trial','Sat','Mass_Immobile','Mass_Mobile','Loc_Ratio']], how = 'left', on = ['Regime','Trial','Sat'])
print(biomass_ratio_df.head())

unsat_sub = biomass_ratio_df[biomass_ratio_df.Sat<1]
unsat_sub["eff_sat"] = unsat_sub.Sat/0.6 - 1/3
marklist = ["o","^","s"]

#%%
fig, axes = plt.subplots(1,2, figsize = (7,3), sharex = True, sharey = True)
sns.scatterplot(data = unsat_sub, x = "eff_sat", y = "State_Ratio", hue = "Regime",  hue_order = ["Slow", "Medium", "Fast"],
                style = "Regime", palette = my_pal, ax = axes.flat[0])
sns.scatterplot(data = unsat_sub, x = "eff_sat", y = "Loc_Ratio", hue = "Regime", hue_order = ["Slow", "Medium", "Fast"],
                style = "Regime",palette = my_pal, ax = axes.flat[1], legend=False)
#axes.flat[0].set_xscale("log")
#axes.flat[0].set_yscale("log")
axes.flat[0].set_title("Ratio of active and\ninactive biomass", fontsize = 12)
axes.flat[1].set_title("Ratio of immobile and\nmobile biomass", fontsize = 12)
axes.flat[0].set_ylabel("Ratio", fontsize = 12)
axes.flat[1].set_ylabel("")
axes.flat[0].set_xlabel("Mean saturation", fontsize = 12)
axes.flat[1].set_xlabel("Mean saturation", fontsize = 12)
axes.flat[0].legend(title="Flow regime", fontsize = 11, title_fontsize = 11)
for a in axes[:]:
    a.tick_params(labelsize = 10)
picname = os.path.join(op_dir,"Fig_5_Unsaturated_fractions_microbes.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
my_pal = {3:"indianred", 2: "g", 0:"steelblue", 1 :"orange"}
path_da_data= os.path.join(uss_dir, "Da_unsaturated.csv")
unsat_da= pd.read_csv(path_da_data)

gvarnames = ["DO", "Nitrate", "DOC", "Ammonium"]#, "Nitrogen", "TOC"]

unsat_data = unsat_da[unsat_da['Chem'].isin (gvarnames)]
unsat_data["logDa"] = np.log10(unsat_data.Da)

unsat_data.loc[unsat_data["logDa"] < -1, "PeDamark"] = 0
unsat_data.loc[(unsat_data["logDa"] > -1) & (unsat_data["logDa"] < 0), "PeDamark"] = 1
unsat_data.loc[(unsat_data["logDa"] > 0) & (unsat_data["logDa"] <0.5), "PeDamark"] = 2
unsat_data.loc[(unsat_data["logDa"] > 0.5), "PeDamark"] = 3

labels = {3 : "log$_{10}$Da > 0.5",
          2 : "0 < log$_{10}$Da < 0.5",
          1 : "-1 < log$_{10}$Da < 0",
         0 : "log$_{10}$Da < -1"}

unsat_data["pc_reldelconc_spatial"] = unsat_data.reldelconc_spatial_fraction * 100

markers = ["o", "s", "d", "^"]
plt.figure()
for frac in [1,2,3]:
    subset = unsat_data[unsat_data['PeDamark'] == frac]
    y = subset["pc_reldelconc_spatial"]
    X = subset[["fraction"]]
    plt.scatter(X*100, y, c = my_pal[frac], marker = markers[frac],alpha = 0.5, label = labels[frac])
plt.xlabel ("Residence time of solutes (%)", **titlekw)
plt.ylabel("Removal of reactive species (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
locs, labels1 = plt.yticks()
#plt.yscale("log")
plt.yticks((0,20,40,60,80,100,120, 140),(0,20,40,60,80,100,120, 140))
plt.xticks((0,20,40,60,80,100),(0,20,40,60,80,100))
plt.tick_params(**labelkw)
plt.legend(title = "Reactive system", title_fontsize = 14, fontsize = 14, loc = (1.05,0.25))
picname = os.path.join(op_dir,"Fig_4_Unsaturated_Da_removal_notblue.png")
#plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
#%%
sat_file = os.path.join(uss_dir,"massflux_comparison_with_sat_26092021.csv")
sat_data = pd.read_csv(sat_file)
sat_data = sat_data.sort_values(by=["Variance", "Anisotropy"])
sat_data["VA"] = sat_data["Variance"].astype(str) + ":" + sat_data["Anisotropy"].astype(int).astype(str)
sat_data["fraction%"] = sat_data.fraction*100

sat_data["eff_sat"] = 1*sat_data["Mean_saturation"]/0.6 -1/3

fig, axmat = plt.subplots(2,1, figsize = (7,8), sharex = True)
axes = axmat.flatten()
sns.boxplot(data = sat_data, y = "eff_sat", x= "VA", hue ="Regime", hue_order = ["Slow", "Medium", "Fast"], palette = my_pal, ax = axes[0])
axes[0].text(-0.2, 0.82, "A", fontsize = 16)
axes[0].axhline(y=0.2, linestyle = ":", c = "grey")
axes[0].text(11, 0.21, "Sr", fontsize = 10)
axes[0].axhline(y=0.8, linestyle = ":", c = "grey")
axes[0].text(11, 0.81, "Smax", fontsize = 10)
axes[0].set_xlabel ("")
axes[0].set_ylabel ("Mean saturation (-)", fontsize = 12)
axes[0].set_ylim((0.1,0.9))
axes[0].legend([], frameon=False)
sns.boxplot(data = sat_data, y = "fraction%", x= "VA", hue ="Regime", hue_order = ["Slow", "Medium", "Fast"], palette = my_pal, ax = axes.flat[1])
axes[1].set_xlabel ("Variance in log permeability field:Anisotropy", fontsize = 12)
axes[1].set_ylabel ("Normalised breakthrough time (%)", fontsize = 12)
axes[1].text(-0.2, 90, "B", fontsize = 16)
xlocs, xlabels = plt.xticks()
plt.xticks(xlocs, ["H", "0.1:2","0.1:5","0.1:10","1:2","1:5","1:10","5:10","5:2","5:5","10:2","10:5","10:10"])
plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tick_params(labelsize = 12)
plt.legend(title = 'Flow regime', fontsize = 12, title_fontsize = 12)
picname = os.path.join(op_dir,"Fig_2_tracer_breakthrough.png")
#plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)
#%%
cv_file = os.path.join(uss_dir,"coeff_var.csv")
cvdata_unsat = pd.read_csv(cv_file)
#Plotting
finaldata_unsat = cvdata_unsat
fig, axes = plt.subplots(4,1, figsize = (4,6), sharex = True, sharey = 'row')
for i,g in zip(list(range(4)),["DOC","DO","Nitrate","Ammonium"]):
    subdata_unsat = finaldata_unsat[(finaldata_unsat.Chem==g)&(finaldata_unsat.Trial!='H')]
    bxplot_unsat = sns.boxplot(x = 'Variance', y = 'cv', hue = 'Regime', hue_order = ["Slow", "Medium", "Fast"], palette = my_pal, 
                data = subdata_unsat, ax = axes[i])
    bxplot_unsat.get_legend().remove()
    axes[i].set_ylabel(g, fontsize = 14)
    axes[i].tick_params(labelsize=14)
    if i<3:
        for a in axes[:]:
            a.set_xlabel(" ")
    else:
        axes[i].set_xlabel("Variance", fontsize = 14)
        a.tick_params(labelsize=14)
plt.subplots_adjust (top = 0.92)
handles, labels = bxplot_unsat.get_legend_handles_labels()
l = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 14, title_fontsize = 14, title = "Flow regime")
picname = os.path.join(op_dir,"Fig_S2_cv_chem.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)

species_text = "Immobile active "
fig, axes = plt.subplots(3,1, figsize = (4,6), sharex = True, sharey = 'row')
for i,g in zip(list(range(3)),["aerobic degraders","nitrate reducers","ammonia oxidizers"]):
    subdata_unsat = finaldata_unsat[(finaldata_unsat.Chem==species_text + g)&(finaldata_unsat.Trial!='H')]
    bxplot_unsat = sns.boxplot(x = 'Variance', y = 'cv', hue = 'Regime', hue_order = ["Slow", "Medium", "Fast"], palette = my_pal, 
                data = subdata_unsat, ax = axes[i])
    bxplot_unsat.get_legend().remove()
    axes[i].set_ylabel(g, fontsize = 14)
    axes[i].tick_params(labelsize=14)
    if i<3:
        for a in axes[:]:
            a.set_xlabel(" ")
    else:
        axes[i].set_xlabel("Variance", fontsize = 14)
        a.tick_params(labelsize=14)
plt.subplots_adjust (top = 0.92)
handles, labels = bxplot_unsat.get_legend_handles_labels()
l = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 14, title_fontsize = 14, title = "Flow regime")
picname = os.path.join(op_dir,"Fig_S7_cv_imm_active.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)

#---EXTRA---###
#%%
fig, axes = plt.subplots(1,2, figsize = (7,3), sharex = True, sharey = True)
sns.scatterplot(data = unsat_sub, x = "eff_sat", y = "Mass_Active", hue = "Regime",  hue_order = ["Fast","Slow", "Medium"],
                style = "Regime", palette = my_pal, ax = axes.flat[0])
sns.scatterplot(data = unsat_sub, x = "eff_sat", y = "Mass_Inactive", hue = "Regime", hue_order = ["Fast","Slow", "Medium"],
                style = "Regime", palette = my_pal, ax = axes.flat[1], legend=False)
#axes.flat[0].set_xscale("log")
#axes.flat[0].set_yscale("log")
axes.flat[0].set_title("Active biomass", fontsize = 12)
axes.flat[1].set_title("Inactive biomass", fontsize = 12)
axes.flat[0].set_ylabel("Mass", fontsize = 12)
axes.flat[1].set_ylabel("")
axes.flat[0].set_xlabel("Mean saturation", fontsize = 12)
axes.flat[1].set_xlabel("Mean saturation", fontsize = 12)
axes.flat[0].legend(title="Flow regime", fontsize = 11, title_fontsize = 11)
axes.flat[0].set_yticks((0.1,1,10),(0.1,1,10))
axes.flat[0].set_xticks((0.45,0.5,0.55,0.6,0.65),(0.45,0.5,0.55,0.6,0.65))
#axes.flat[1].set_xticks((0.45,0.5,0.55,0.6,0.65),(0.45,0.5,0.55,0.6,0.65))
for a in axes[:]:
    a.tick_params(labelsize = 10)
#picname = os.path.join(op_dir,"Fig_4_5_Unsaturated_fractions_microbes.png")
#plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
fig, axes = plt.subplots(1,2, figsize = (7,3), sharex = True, sharey = True)
sns.scatterplot(data = unsat_sub, x = "eff_sat", y = "Mass_Immobile", hue = "Regime",  hue_order = ["Fast","Slow", "Medium"],
                style = "Regime", palette = my_pal, ax = axes.flat[0])
sns.scatterplot(data = unsat_sub, x = "eff_sat", y = "Mass_Mobile", hue = "Regime", hue_order = ["Fast","Slow", "Medium"],
                style = "Regime", palette = my_pal, ax = axes.flat[1], legend=False)
#axes.flat[0].set_xscale("log")
#axes.flat[0].set_yscale("log")
axes.flat[0].set_title("Immobile biomass", fontsize = 12)
axes.flat[1].set_title("Mobile biomass", fontsize = 12)
axes.flat[0].set_ylabel("Mass", fontsize = 12)
axes.flat[1].set_ylabel("")
axes.flat[0].set_xlabel("Mean saturation", fontsize = 12)
axes.flat[1].set_xlabel("Mean saturation", fontsize = 12)
axes.flat[0].legend(title="Flow regime", fontsize = 11, title_fontsize = 11)
axes.flat[0].set_yticks((0.1,1,10),(0.1,1,10))
axes.flat[0].set_xticks((0.45,0.5,0.55,0.6,0.65),(0.45,0.5,0.55,0.6,0.65))
#axes.flat[1].set_xticks((0.45,0.5,0.55,0.6,0.65),(0.45,0.5,0.55,0.6,0.65))
for a in axes[:]:
    a.tick_params(labelsize = 10)
#picname = os.path.join(op_dir,"Fig_4_5_Unsaturated_fractions_microbes.png")
#plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

my_pal = {3:"indianred", 2: "g", 0:"steelblue", 1 :"orange"}
path_da_data= os.path.join(uss_dir, "Da_unsaturated.csv")
unsat_da= pd.read_csv(path_da_data)

gvarnames = ["DO", "Nitrate", "DOC", "Ammonium"]#, "Nitrogen", "TOC"]

unsat_data = unsat_da[unsat_da['Chem'].isin (gvarnames)]
unsat_data["logDa"] = np.log10(unsat_data.Da)

unsat_data.loc[unsat_data["logDa"] < -1, "PeDamark"] = 0
unsat_data.loc[(unsat_data["logDa"] > -1) & (unsat_data["logDa"] < 0), "PeDamark"] = 1
unsat_data.loc[(unsat_data["logDa"] > 0) & (unsat_data["logDa"] <0.5), "PeDamark"] = 2
unsat_data.loc[(unsat_data["logDa"] > 0.5), "PeDamark"] = 3

labels = {3 : "log$_{10}$Da > 0.5",
          2 : "0 < log$_{10}$Da < 0.5",
          1 : "-1 < log$_{10}$Da < 0",
         0 : "log$_{10}$Da < -1"}

unsat_data["pc_reldelconc_spatial"] = unsat_data.reldelconc_spatial_fraction * 100

markers = ["o", "s", "d", "^"]
plt.figure()
for frac in [0,1,2,3]:
    subset = unsat_data[unsat_data['PeDamark'] == frac]
    y = subset["pc_reldelconc_spatial"]
    X = subset[["fraction"]]
    plt.scatter(X*100, y, c = my_pal[frac], marker = markers[frac],alpha = 0.5, label = labels[frac])
plt.xlabel ("Residence time of solutes (%)", **titlekw)
plt.ylabel("Removal of reactive species (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
locs, labels1 = plt.yticks()
plt.yscale("log")
plt.tick_params(**labelkw)
plt.legend(title = "Reactive system", title_fontsize = 14, fontsize = 14, loc = (1.05,0.25))
picname = os.path.join(op_dir,"Fig_8_Unsaturated_Da_removal_all.png")
#plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
