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
import seaborn as sns
#User libraries
from DS.data_reader import data_processing as proc
from DS.data_reader.data_processing import tracerstudies
import DS.analyses.transient as ta

#%%
## directories for personal computer:
uss_dir = r"C:\Users\swami\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
op_dir = r"C:\Users\swami\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"

#%%
## directories for work computer
uss_dir = r"C:\Users\swkh9804\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
op_dir = r"C:\Users\swkh9804\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
#%%
#Standard color and font options
my_pal = {2:"indianred", 11:"g", 22:"steelblue", "DO":"indianred", "Nitrate":"g", "Ammonium":"steelblue",
          'Slow':"indianred", "Medium":"g", "Fast":"steelblue", 0: "steelblue", 1: "orange", 2: "g",
          3:"indianred"}
my_style = {'Slow':"o", "Medium":"^", "Fast":"s"}

marklist = ["o", "s", "^","d"]
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

unbio_file = os.path.join(uss_dir,"biomass_comparison_09082022.csv")#biomass_comparison_03082021.csv")
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

path_chem_data= os.path.join(uss_dir, "conc_comparison_09082022.csv")
unsat_conc_data= pd.read_csv(path_chem_data)

path_chem_data= os.path.join(uss_dir, "massflux_comparison_09082022.csv")
unsat_mf_data= pd.read_csv(path_chem_data)

#%%
da_hom = unsat_conc_data[unsat_conc_data.Trial == "H"]
print (da_hom.shape)
plt.figure()
sns.barplot(x = 'Chem', y = 'reldelconc', hue = 'Regime', hue_order= ['Slow', 'Medium','Fast'],
palette= my_pal, data = da_hom[da_hom.Chem != "DO"])
plt.xlabel ("Chemical species", **titlekw)
plt.ylabel("Relative removal (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
plt.tick_params(**labelkw, rotation = 30)
plt.legend(title = "Flow regime", title_fontsize = 14, fontsize = 14, loc = (1.05,0.25))
picname = os.path.join(op_dir,"Fig_1_Chem_removal.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)

#%%
sat_file = os.path.join(uss_dir,"massflux_comparison_09082022.csv")
sat_data = pd.read_csv(sat_file)
sat_data = sat_data.sort_values(by=["Variance", "Anisotropy"])
sat_data["VA"] = sat_data["Variance"].astype(str) + ":" + sat_data["Anisotropy"].astype(int).astype(str)
sat_data["fraction%"] = sat_data.fraction*100

sat_data["eff_sat"] = sat_data["Mean_saturation"]

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
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)
#%%
diff_data_path = os.path.join(uss_dir, "aero_rates_09082022.csv")
mdiff_data = pd.read_csv (diff_data_path)
print (mdiff_data.columns)
print(mdiff_data.rate.unique())
aero_data = pd.merge(unsat_conc_data, mdiff_data[['Trial', 'Regime', 'rate', 'rate_val', 'rate_spatial_fraction']], on = ['Trial', 'Regime'])
aero_data['rate_spatial_fraction%'] = aero_data['rate_spatial_fraction']*100
aero_data['fraction%'] = aero_data['fraction']*100
diff_data = mdiff_data[mdiff_data['rate']=='aeration'].reset_index()
diff_data.rename(columns={"rate_val": "aeration"}, inplace=True)
resp_data = mdiff_data[mdiff_data['rate']=='Total_respiration'].reset_index()
resp_data.rename(columns={"rate_val": "Total_resp"}, inplace=True)
resp_aer_data = pd.merge(diff_data, resp_data[['Regime', 'Trial', 'Total_resp']], on = ['Regime', 'Trial'])
resp_aer_data["resp_diff"] = resp_aer_data.Total_resp/resp_aer_data.aeration

for r in Regimes:
    for t in list(resp_aer_data.Trial.unique()):
        for c in ['aeration']:
            spat_n_base = resp_aer_data.loc[(resp_aer_data.Regime == r) & (resp_aer_data.rate == c) & (resp_aer_data.Trial == 'H')]['resp_diff'].values[0]
            resp_aer_data.loc[(resp_aer_data.Regime == r) & (resp_aer_data.rate == c) & (resp_aer_data.Trial == t), 'spatial_base'] = spat_n_base

resp_aer_data['resp_sub_spatial_fraction'] = resp_aer_data['resp_diff']/resp_aer_data['spatial_base']
resp_aer_data['resp_sub_spatial_fraction%'] = resp_aer_data['resp_sub_spatial_fraction'] * 100
resp_aer_data['fraction%'] = resp_aer_data['fraction']*100
#%%
fig, a = plt.subplots(1,3, figsize = (9,3), sharex = 'all', sharey = 'all')
a[0].text(s="A", x = 30, y = 900, **titlekw)
a[1].text(s="B", x = 30, y = 900, **titlekw)
a[2].text(s="C", x = 30, y = 900, **titlekw)
a[0].set_title("DO consumption\nrate",  **titlekw)
a[1].set_title("Aeration\nrate",  **titlekw)
a[2].set_title(r"$\frac{Respiration}{Diffusion}$",  **titlekw)
a[0].set_ylabel("Normalized with\nbase case (%)",  **titlekw)

sns.scatterplot(x = 'fraction%', y =  'rate_spatial_fraction%', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data[aero_data.rate=='Total_respiration'], ax = a[0])
sns.scatterplot(x = 'fraction%', y = 'rate_spatial_fraction%', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data[aero_data.rate=='aeration'], ax = a[1])
sns.scatterplot(x = 'fraction%', y = 'resp_sub_spatial_fraction%', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = resp_aer_data, ax = a[2])
for all in a.flatten():
    all.tick_params(**labelkw)
    all.legend().remove()
    all.set_xlabel ("Residence time\nof solutes (%)", **titlekw)                 
for ax in a[:]:
    ax.tick_params(**labelkw)
    ax.set_yscale("log")
plt.legend(title = "Flow regime", title_fontsize = 14, fontsize = 14, loc = (-1.75,-0.7), ncol = 3)
picname = os.path.join(op_dir,"Fig_3_respiration_aeration.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)
picname = os.path.join(op_dir,"Fig_3_respiration_aeration.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)

#%%
r_d_unsat_data = pd.merge(resp_aer_data[['Regime','Trial', 'resp_diff']], unsat_conc_data, on = ['Regime','Trial']).reset_index()
r_d_unsat_data['fraction%'] = r_d_unsat_data.fraction*100
r_d_unsat_data['pc_reldelconc_spatial'] = r_d_unsat_data.reldelconc_spatial_fraction*100
r_d_unsat_data.loc[r_d_unsat_data['resp_diff']>=10, "diff"] = int(0)
r_d_unsat_data.loc[(r_d_unsat_data['resp_diff']>=1) & (r_d_unsat_data['resp_diff']<10), "diff"] = int(1)
r_d_unsat_data.loc[(r_d_unsat_data['resp_diff']<1), "diff"] = int(2)
grey_tri = mlines.Line2D([], [], linestyle = '', marker = "^", markerfacecolor = "grey", markeredgecolor = "grey", markersize=10, label='Nitrate', alpha = 0.5)
grey_dot = mlines.Line2D([], [], linestyle = '', marker = "o", markerfacecolor = "grey", markeredgecolor = "grey", markersize=10, label='DOC', alpha = 0.5)
grey_square = mlines.Line2D([], [], linestyle = '', marker = "s", markerfacecolor = "grey", markeredgecolor = "grey",markersize=10, label='Ammonium', alpha = 0.5)
blue_patch = mpatches.Patch(color="steelblue", label= 'Ratio>10', alpha = 0.5)
orange_patch = mpatches.Patch(color = "orange", label =  '1<Ratio<10', alpha = 0.5)
red_patch = mpatches.Patch(color="green", label= 'Ratio~1', alpha = 0.5)
chem_leg_list = [grey_tri, grey_dot, grey_square]
patchlist = [blue_patch, orange_patch, red_patch]
fig, axes = plt.subplots(2,1,figsize=[6, 8], sharex = True)
ax = axes[0]
axins = axes[1]
ax.text(s="A", x = 30, y = 100000, **titlekw)
axins.text(s="B", x = 30, y = 180, **titlekw)
#a[0,1].text(s="B", x = 30, y = 140, **titlekw)
for c in gvarnames:
    subset_1 = r_d_unsat_data[r_d_unsat_data['Chem'] == c]
    frac = gvarnames.index(c)
    for d_frac in [0,1,2]:
        subset = subset_1[subset_1['diff']==d_frac]
        if subset.shape[0]<1:
            pass                          
        else:
            y = subset["pc_reldelconc_spatial"]
            X = subset[["fraction"]]
            ax.scatter(X*100, y, c = my_pal[d_frac], marker = marklist[frac],alpha = 0.5, label = c)
ax.vlines(x = 30, ymin = 10, ymax = 200, linestyles = 'dashed', colors = 'grey')
ax.vlines(x = 102, ymin = 10, ymax = 200, linestyles = 'dashed', colors = 'grey')
ax.hlines(y = 200, xmin = 30, xmax = 102, linestyles = 'dashed', colors = 'grey')
ax.hlines(y = 10, xmin = 30, xmax = 102, linestyles = 'dashed', colors = 'grey')
ax.set_yscale("log")
ax.tick_params(**labelkw)
ax.set_ylabel("Removal of reactive\nspecies (%)", ha='center', va='center', rotation='vertical', labelpad = 12, **titlekw)
for c in gvarnames:
    subset_1 = r_d_unsat_data[r_d_unsat_data['Chem'] == c]
    frac = gvarnames.index(c)
    for d_frac in [0,1,2]:
        subset = subset_1[subset_1['diff']==d_frac]
        if subset.shape[0]<1:
            pass                          
        else:
            y = subset["pc_reldelconc_spatial"]
            X = subset[["fraction"]]
            axins.scatter(X*100, y, c = my_pal[d_frac], marker = marklist[frac],alpha = 0.5, label = c)
axins.set_ylim(10,200)
plt.tick_params(**labelkw)
axins.set_ylabel("Removal of reactive\nspecies (%)", ha='center', va='center', rotation='vertical', labelpad = 15, **titlekw)
axins.set_xlabel ("Residence time of solutes (%)", **titlekw)
plt.tick_params(**labelkw)
legend_flow = plt.legend(handles=chem_leg_list, ncol = 3,
        bbox_to_anchor=(0.5, -0.35),
        loc="center",
        title="Chemical",
        fontsize=14, title_fontsize = 14)
plt.legend(handles=patchlist, ncol = 3,
        bbox_to_anchor=(0.5, -0.62),
        loc="center",
        title=r"$\frac{Respiration}{Diffusion}$",
        fontsize=14, title_fontsize = 14)
plt.gca().add_artist(legend_flow)
picname = os.path.join(op_dir,"Fig_4_Unsaturated_chem_removal.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = os.path.join(op_dir,"Fig_4_Unsaturated_chem_removal.pdf")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
#%%
fig, axes = plt.subplots(1,2, figsize = (7,3), sharex = True, sharey = True)
sns.scatterplot(data = unsat_sub, x = "eff_sat", y = "State_Ratio", hue = "Regime",  hue_order = ["Slow", "Medium", "Fast"],
                style = "Regime", palette = my_pal, ax = axes.flat[0])
sns.scatterplot(data = unsat_sub, x = "eff_sat", y = "Loc_Ratio", hue = "Regime", hue_order = ["Slow", "Medium", "Fast"],
                style = "Regime",palette = my_pal, ax = axes.flat[1], legend=False)
axes.flat[0].set_title("Ratio of active and\ninactive biomass", **titlekw)
axes.flat[1].set_title("Ratio of immobile and\nmobile biomass", **titlekw)
axes.flat[0].set_ylabel("Ratio", **titlekw)
axes.flat[1].set_ylabel("")
axes.flat[0].set_xlabel("Mean saturation", **titlekw)
axes.flat[1].set_xlabel("Mean saturation", **titlekw)
axes.flat[0].text(s="A", x = 0.4, y = 9, **titlekw)
axes.flat[1].text(s="B", x = 0.4, y = 9, **titlekw)
#axes.flat[1].tick_params(**labelkw)
#axes.flat[0].tick_params(**labelkw)
axes.flat[0].legend(title="Flow regime", fontsize = 14, title_fontsize = 14, loc = (0.33, -0.55), ncol = 3)
for a in axes[:]:
    a.tick_params(labelsize = 12)
picname = os.path.join(op_dir,"Fig_5_Unsaturated_fractions_microbes.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
#Figure S1: 1D profile dissolved species
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
dashedline = mlines.Line2D([], [], linestyle = '-.', color='grey', markersize=15, label='Homogeneous')
solidline = mlines.Line2D([], [], linestyle = 'solid', color='grey', markersize=15, label='Heterogeneous')
blue_patch = mpatches.Patch(color="blue", label= 'Ammonium', alpha = 0.5)
red_patch = mpatches.Patch(color = "red", label =  'DO', alpha = 0.5)
black_patch = mpatches.Patch(color="black", label= 'DOC', alpha = 0.5)
green_patch = mpatches.Patch(color="darkgreen", label='Nitrate', alpha = 0.5)
patchlist = [blue_patch, green_patch, red_patch, black_patch, dashedline, solidline]
legendsize = 16
axissize = 16
ticksize = 14
titlesize = 20
yin = 6
yout = -7
xleft = 0
xright = -1
vertnodes = 113
velem = 0.005
vedge = 0.0025
d = r"D:\Data\Richards_flow\RF_big_sat_2"
Regimes = ["Slow", "Medium", "Fast"]
trialist = proc.masterscenarios("Unsaturated")
Trial = ["50", "84", "118"]
species = proc.speciesdict("Unsaturated")
gvarnames = ["DO", "DOC", "Ammonium", "Nitrate"]
cvars = list(species[g]['TecIndex'] for g in gvarnames)
velindex = 2
colors = ["red", "black", "blue", "darkgreen"]
columntitles = ["Slow flow", "Medium flow", "Fast flow"]
pad = 230
i = 0
figbig, axes = plt.subplots(3,3, figsize=(13, 10), sharey = True, sharex = True)
for r in Regimes:
    fileh = os.path.join(d, r + "AR_0_RF-AH_df.npy")
    datah = np.load(fileh)
    conctimeh, TotalFlowh, Headinlettimeh = ta.conc_time(datah, 0, -1, xleft, xright, vertnodes, gvarnames, "Unsaturated", element_length = velem, edge_length = vedge)
    for t in Trial:
        ax = axes[Trial.index(t),Regimes.index(r)]
        host = ax
        file = os.path.join(d, r + "AR_0_RF-A" + str(t) + "_df.npy")
        data = np.load(file)
        conctime, TotalFlow, Headinlettime = ta.conc_time(data, 0, -1, xleft, xright, vertnodes, gvarnames, "Unsaturated", element_length = velem, edge_length = vedge)
        yindex = list(range(np.shape(conctime)[1]-12))
        host.plot(conctimeh[-1, yin:yout+1, 0],yindex,label=gvarnames[0],color=colors[0],linestyle="-.")
        host.plot(conctime[-1, yin:yout+1, 0],yindex,label=gvarnames[0],color=colors[0],linestyle="-")
        host.plot(conctimeh[-1, yin:yout+1, 1],yindex,label=gvarnames[1],color=colors[1],linestyle="-.")
        host.plot(conctime[-1, yin:yout+1, 1],yindex,label=gvarnames[1],color=colors[1],linestyle="-",)
        par1 = host.twiny()
        par2 = host.twiny()
        # Offset the top spine of par2.  The ticks and label have already been
        # placed on the top by twiny above.
        par2.spines["top"].set_position(("axes", 1.2))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        make_patch_spines_invisible(par2)
        # Second, show the right spine.
        par1.plot(conctimeh[-1, yin:yout+1, 2],yindex,label=gvarnames[2],color=colors[2],linestyle="-.")
        par1.plot(conctime[-1, yin:yout+1, 2],yindex,label=gvarnames[2],color=colors[2],linestyle="-")
        par2.plot(conctimeh[-1, yin:yout+1, 3],yindex,label=gvarnames[3],color=colors[3],linestyle="-.")
        par2.plot(conctime[-1, yin:yout+1, 3],yindex,label=gvarnames[3],color=colors[3],linestyle="-")
        host.set_ylim(0, 51)
        host.set_xlim(0, 800)
        par1.set_xlim(20, 60)
        par2.set_xlim(50, 260)
        host.xaxis.label.set_color("black")
        tkw = dict(size=4, width=1.5, labelsize=ticksize)
        host.tick_params(axis="x", colors="black", **tkw)
        host.tick_params(axis="y", **tkw)
        if Trial.index(t)==0:
            host.set_title (r + " flow", fontsize = axissize)
            par2.spines["top"].set_visible(True)
            par1.xaxis.label.set_color("blue") 
            par2.xaxis.label.set_color("darkgreen")
            par1.tick_params(axis="x", colors="blue", **tkw)
            par2.tick_params(axis="x", colors="darkgreen", **tkw)
            par1.set_xlabel(str(gvarnames[2]) + " (uM)", fontsize=axissize)
            par2.set_xlabel(str(gvarnames[3]) + " (uM)", fontsize=axissize)
        elif Trial.index(t)==len(Trial)-1:
            host.set_xlabel("DOC, DO (uM)", fontsize=axissize)
            par1.set_xticks([])
            par2.set_xticks([])
        else:
            par1.set_xticks([])
            par2.set_xticks([])
        i+=1
figbig.gca().invert_yaxis()
figbig.subplots_adjust(top=1.0, hspace = 0.2, wspace = 0.2)
for t,a in zip(Trial[::-1],range(3)):
    plt.annotate("Variance: " + str(trialist[t]["Het"])+ " &\nAnisotropy: " + str(trialist[t]["Anis"]),
                 xy=(0.1, 0.17), xytext=(-58, 0.6 + pad*a),
                xycoords='figure fraction', textcoords='offset points',
                rotation = "vertical",
                size='large', ha='center', va='baseline',
                fontsize = 16)
    axes.flat[3*a].set_ylabel("Y (cm)", fontsize=axissize)
plt.legend(handles = patchlist, ncol = 3, fontsize = legendsize,
           bbox_to_anchor = (-0.2,-0.6),
           loc = 'lower right')

picname = os.path.join(op_dir, "FigureS1_dissolved_species_1D.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = os.path.join(op_dir, "FigureS1_dissolved_species_1D.pdf")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
## Figure S2: 1D profile biomass
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
dashedline = mlines.Line2D([], [], linestyle = '-.', color='grey', markersize=15, label='Homogeneous')
solidline = mlines.Line2D([], [], linestyle = 'solid', color='grey', markersize=15, label='Heterogeneous')
blue_patch = mpatches.Patch(color="blue", label= 'Ammonia oxidizers', alpha = 0.5)
black_patch = mpatches.Patch(color="black", label= 'Aerobes', alpha = 0.5)
green_patch = mpatches.Patch(color="darkgreen", label='Nitrate reducers', alpha = 0.5)
patchlist = [blue_patch, dashedline, black_patch, solidline, green_patch]
legendsize = 16
axissize = 16
ticksize = 14
titlesize = 20
yin = 6
yout = -7
xleft = 0
xright = -1
vertnodes = 113
velem = 0.005
vedge = 0.0025
d = r"D:\Data\Richards_flow\RF_big_sat_2"
Regimes = ["Slow", "Medium", "Fast"]
trialist = proc.masterscenarios("Unsaturated")
Trial = ["50", "84", "118"]
species = proc.speciesdict("Unsaturated")
Regimes = ["Slow", "Medium", "Fast"]
gvarnames = list(g for g in species.keys() if (species[g]["State"] == "Active") and (species[g]["Location"] == "Immobile"))
gvarnames.remove('Immobile active sulphate reducers')
cvars = list(species[g]['TecIndex'] for g in gvarnames)
velindex = 2
colors = ["black", "darkgreen", "blue"]
columntitles = ["Slow flow", "Medium flow", "Fast flow"]
pad = 230
i = 0
figbig, axes = plt.subplots(3,3, figsize=(13, 10), sharey = True, sharex = True)
for r in Regimes:
    fileh = os.path.join(d, r + "AR_0_RF-AH_df.npy")
    datah = np.load(fileh)
    masstimeh, conctimeh = ta.biomasstimefunc(datah,yin,yout,xleft,xright, vertnodes, gvarnames,"Unsaturated")
    for t in Trial:    
        host = axes[Trial.index(t),Regimes.index(r)]
        #i = Trial.index(t)*len(Regimes) + Regimes.index(r)
        file =os.path.join(d, r + "AR_0_RF-A" + str(t) + "_df.npy")
        data = np.load(file)
        DIR = d + "/" + r + "AR_0/"
        masstime, conctime = ta.biomasstimefunc(data,yin,yout,xleft,xright, vertnodes, gvarnames,"Unsaturated")
        yindex = list(range(np.shape(conctime)[1]-12))
        #fig, host = axe.subplots()
        host.plot(conctimeh[-1, yin:yout+1, 0],yindex,label=gvarnames[0],color=colors[0],linestyle="-.")
        host.plot(conctime[-1, yin:yout+1, 0],yindex,label=gvarnames[0],color=colors[0],linestyle="-")
        par1 = host.twiny()
        par2 = host.twiny()

        # Offset the top spine of par2.  The ticks and label have already been
        # placed on the top by twiny above.
        par2.spines["top"].set_position(("axes", 1.2))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        make_patch_spines_invisible(par2)
        # Second, show the right spine.

        par1.plot(conctimeh[-1, yin:yout+1, 1],yindex,label=gvarnames[1],color=colors[1],linestyle="-.")
        par1.plot(conctime[-1, yin:yout+1, 1],yindex,label=gvarnames[1],color=colors[1],linestyle="-")
        par2.plot(conctimeh[-1, yin:yout+1, 2],yindex,label=gvarnames[2],color=colors[2],linestyle="-.")
        par2.plot(conctime[-1, yin:yout+1, 2],yindex,label=gvarnames[2],color=colors[2],linestyle="-")

        host.set_ylim(0, 51)
        host.set_xlim(0, 500)
        par1.set_xlim(0, 200)
        par2.set_xlim(0, 30)
        host.xaxis.label.set_color("black")
        tkw = dict(size=4, width=1.5, labelsize=ticksize)
        host.tick_params(axis="x", colors="black", **tkw)
        host.tick_params(axis="y", **tkw)
        if Trial.index(t)==0:
            host.set_title (r + " flow", fontsize = axissize)
            par2.spines["top"].set_visible(True)
            par1.xaxis.label.set_color("blue")
            par2.xaxis.label.set_color("darkgreen")
            par1.tick_params(axis="x", colors="blue", **tkw)
            par2.tick_params(axis="x", colors="darkgreen", **tkw)
            par1.set_xlabel(species[gvarnames[1]]["Graphname"] + " (uM)", fontsize=axissize)
            par2.set_xlabel(species[gvarnames[2]]["Graphname"] + " (uM)", fontsize=axissize)
        elif Trial.index(t)==len(Trial)-1:
            host.set_xlabel(species[gvarnames[0]]["Graphname"], fontsize=axissize)
            par1.set_xticks([])
            par2.set_xticks([])
        else:
            par1.set_xticks([])
            par2.set_xticks([])
        i+=1
figbig.gca().invert_yaxis()
figbig.subplots_adjust(top=1.0, hspace = 0.2, wspace = 0.2)
for t,a in zip(Trial[::-1],range(3)):
    plt.annotate("Variance: " + str(trialist[t]["Het"])+ " &\nAnisotropy: " + str(trialist[t]["Anis"]),
                 xy=(0.1, 0.17), xytext=(-60, 0.8 + pad*a),xycoords='figure fraction',textcoords='offset points',rotation = "vertical",size='large', ha='center', va='baseline',fontsize = 16)
    axes.flat[3*a].set_ylabel("Y (cm)", fontsize=axissize)
plt.legend(handles = patchlist, ncol = 3, fontsize = legendsize,bbox_to_anchor = (-0.6,-0.6),loc = 'lower center')

picname = os.path.join(op_dir, "FigureS2_biomass_1D.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = os.path.join(op_dir, "FigureS2_biomass_1D.pdf")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
## Figure S3: Heatmap of biomass contributors
unbio_data['Contribution_pc'] = unbio_data.Biomass_contribution*100
base_bio =  unbio_data[unbio_data.Trial=='H']
base_bio_pivot = pd.pivot(data = base_bio[['Regime', 'Chem', 'Contribution_pc']], index = ['Chem'], columns = ['Regime'], values = 'Contribution_pc')
sns.heatmap(base_bio_pivot, cmap = 'YlGnBu')
plt.xlabel ("Flow regime")
plt.ylabel ("Microbial group")
picname = os.path.join(op_dir, "FigureS3_biomass_heatmap.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = os.path.join(op_dir, "FigureS3_biomass_heatmap.pdf")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
## Figure S4: 2D saturation and velocity heatmap
import matplotlib.gridspec as gridspec
Regimes = ["Slow", "Medium", "Fast"]
trialist = proc.masterscenarios("Unsaturated")
Trial = ["50", "72", "118"]
species = proc.speciesdict("Unsaturated")
gvarnames = ["Sat"]
gindx = 4
velindex = 2
yin = 6
yout = -7
colorscheme = 'YlGnBu'
columntitles = ["Velocity\ndistribution pattern", "Slow\nflow", "Medium\nflow", "Fast\nflow"]
fig, axes = plt.subplots(3,4, figsize=(10, 10))
pad = 210
i = 0
for t in Trial:
    file = os.path.join(d, "MediumAR_0_RF-A" + str(t)+ "_df.npy")
    data = np.load(file)
    axe = axes[Trial.index(t),0]
    velocity = abs(data[velindex, -1, yin:yout+1, :])
    sns.heatmap(velocity, cmap = colorscheme, ax = axe, cbar = False)
    axe.set_ylabel ("Variance: " + str(trialist[t]["Het"])+ " &\nAnisotropy: " + str(trialist[t]["Anis"]),
                       rotation = "vertical", fontsize = 16, ha = "center")
    if i ==0:
        axe.set_title("Velocity distribution\npattern", fontsize = 16, ha = "center")
    axe.set_xticks([])
    axe.set_yticks([])
    fig.add_subplot(axe)    
    for r in Regimes:
        axer = axes[Trial.index(t),Regimes.index(r)+1]
        file = os.path.join(d, r + "AR_0_RF-A" + str(t) + "_df.npy")
        data = np.load(file)
        sns.heatmap (data[4, -1, yin:yout+1, :], cmap = colorscheme, ax= axer)
        if Trial.index(t)==0:
            axer.set_title(r + "\nflow", fontsize = 16, ha = "center")
        axer.set_xticks([])
        axer.set_yticks([])
picname = os.path.join(op_dir, "FigureS4_saturation_heatmap.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = os.path.join(op_dir, "FigureS4_saturation_heatmap.pdf")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
#%%
#Figure S6: Distribution of dissolved species heatmap
import matplotlib.gridspec as gridspec

legendsize = 16
axissize = 16
ticksize = 14
titlesize = 20
yin = 6
yout = -7
xleft = 0
xright = -1
vertnodes = 113
velem = 0.005
vedge = 0.0025
d = r"D:\Data\Richards_flow\RF_big_sat_2"
Regimes = ["Slow", "Medium", "Fast"]
trialist = proc.masterscenarios("Unsaturated")
Trial = ["50", "84", "118"]
species = proc.speciesdict("Unsaturated")
gvarnames = ["DO", "DOC", "Ammonium", "Nitrate"]
velindex = 2
colorscheme = 'YlGnBu'
columntitles = ["Velocity\ndistribution pattern", "Slow\nflow", "Medium\nflow", "Fast\nflow"]
fig = plt.figure(figsize=(14, 14))
outer = gridspec.GridSpec(3, 4, wspace=0.2, hspace=0.2)
pad = 210
for t in Trial:
    file = os.path.join(d, "MediumAR_0_RF-A"+str(t)+"_df.npy")
    data = np.load(file)
    left = gridspec.GridSpecFromSubplotSpec(1, 1,
                subplot_spec=outer[4*Trial.index(t)], wspace=0.3, hspace=0.1)
    axe = plt.Subplot(fig, left[0])
    velocity = abs(data[velindex, -1, yin:yout+1, :])
    sns.heatmap(velocity, cmap = colorscheme, ax = axe, cbar = False)
    axe.set_ylabel ("Variance: " + str(trialist[t]["Het"])+ " &\nAnisotropy: " + str(trialist[t]["Anis"]),
                   rotation = "vertical", fontsize = 16, ha = "center")
    axe.set_xticks([])
    axe.set_yticks([])
    fig.add_subplot(axe)
    for r in Regimes:
        i = Trial.index(t)*len(Regimes) + Regimes.index(r) + Trial.index(t) + 1
        if i%4 != 0:
           inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                    subplot_spec=outer[i], wspace=0.4, hspace=0.15)
           file = os.path.join(d, r+"AR_0_RF-A"+str(t)+"_df.npy")
           data = np.load(file)
           for g in gvarnames:
                axe = plt.Subplot(fig, inner[gvarnames.index(g)])
                sns.heatmap (data[species[g]["TecIndex"], -1, yin:yout+1, :], cmap = colorscheme, ax= axe)
                axe.set_title(g, fontsize = 13, ha = "center")
                axe.set_xticks([])
                axe.set_yticks([])
                fig.add_subplot(axe)
for a in list(range(4)):
    plt.annotate(columntitles[a], xy=(0.001, 0), ha='center', xytext=(-10-pad*(3-a), 700),
                textcoords='offset points', fontsize = 16)
picname = os.path.join(op_dir, "FigureS6_dissolved_species_heatmaps.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = os.path.join(op_dir, "FigureS6_dissolved_species_heatmaps.pdf")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
#Figure S7: Aeration rates
plt.figure()
plt.title("Aeration rate",  **titlekw)
plt.ylabel("Rate (uM d-1)",  **titlekw)
sns.scatterplot(x = 'fraction%', y = 'rate_val', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data[aero_data.rate=='aeration'])
plt.xlabel ("Residence time\nof solutes (%)", **titlekw)                 
plt.legend(title = "Flow regime", title_fontsize = 14, fontsize = 14, loc = (0.05,-0.5), ncol = 3)
picname = os.path.join(op_dir,"Fig_S7_aeration.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)
picname = os.path.join(op_dir,"Fig_S7_aeration.pdf")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)

#%%
#Figure S8: Distribution of biomass heatmap
import matplotlib.gridspec as gridspec
d = r"D:\Data\Richards_flow\RF_big_sat_2"
Regimes = ["Slow", "Medium", "Fast"]
trialist = proc.masterscenarios("Unsaturated")
Trial = ["50", "84", "118"]
species = proc.speciesdict("Unsaturated")
iaspecies = list(g for g in species if ((species[g]["State"]=="Active") and (species[g]["Location"] == "Immobile")))
gvarnames = list(g for g in iaspecies if (g != "Immobile active sulphate reducers"))
sptitles = ["Aerobic\ndegraders", "Nitrate\nreducers", "Ammonia\noxidizers"]
velindex = 2
colorscheme = 'YlGnBu'
columntitles = ["Velocity\ndistribution pattern", "Slow flow", "Medium flow", "Fast flow"]
fig = plt.figure(figsize=(24, 8))
outer = gridspec.GridSpec(3, 4, width_ratios = [0.2,1, 1, 1],wspace=0.15, hspace=0.3)
pad = 400
for t in Trial:
    file = os.path.join(d, "MediumAR_0_RF-A"+str(t)+ "_df.npy")
    data = np.load(file)
    left = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[4*Trial.index(t)],wspace=0.3,hspace=0.1)
    axe=plt.Subplot(fig, left[0])
    velocity=abs(data[velindex,-1,yin:yout+1,:])
    sns.heatmap(velocity,cmap=colorscheme,ax=axe,cbar=False)
    axe.set_ylabel ("Variance: "+str(trialist[t]["Het"])+" &\nAnisotropy: "+str(trialist[t]["Anis"]),
                   rotation="vertical",fontsize=18,ha="center")
    axe.set_xticks([])
    axe.set_yticks([])
    fig.add_subplot(axe)
    for r in Regimes:
        i=Trial.index(t)*len(Regimes)+Regimes.index(r)+Trial.index(t)+1
        if i%4!=0:
            inner = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[i],wspace=0.3,hspace=0.1)
            file = os.path.join(d, r+"AR_0_RF-A"+str(t)+ "_df.npy")
            data = np.load(file)
            for g in gvarnames:
                axe=plt.Subplot(fig,inner[gvarnames.index(g)])
                sns.heatmap (data[species[g]["TecIndex"],-1,yin:yout+1,:],cmap=colorscheme,ax= axe)
                axe.set_title(sptitles[gvarnames.index(g)],fontsize=13,ha="center")
                axe.set_xticks([])
                axe.set_yticks([])
                fig.add_subplot(axe)
for a in range(1,4,1):
    plt.annotate(columntitles[a], xy=(0.001, 0), ha='center', xytext=(-100-pad*(3-a), 370),
                textcoords='offset points', fontsize = 18)
plt.annotate(columntitles[0],xy=(0.1, 0.92),xytext=(0.1, 0.85),xycoords='figure fraction',textcoords='figure fraction',size='large',ha='center',va='baseline',fontsize=18)
picname = os.path.join(op_dir, "FigureS8_immobile_biomass_heatmaps.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = os.path.join(op_dir, "FigureS8_immobile_biomass_heatmaps.pdf")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
#Figure S9: Microbial biomass fractions with spatial heterogeneity
#Load standard values required to navigate through the datasets
## WIP
#%%
#Supplementary figures on coeefcient of variation
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
picname = os.path.join(op_dir,"Fig_SXX_cv_chem.png")
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
picname = os.path.join(op_dir,"Fig_SXX_cv_imm_active.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)
