"""
This is a script file to generate graphics for respiration diffusion related
visualizations.
"""
#%%
#Native libraries
import os

#Third party processing libraries
import pandas as pd

#Visualisation libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.patches as mpatches

#User libraries
from DS.data_reader import data_processing as proc

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

path_da_data= os.path.join(uss_dir, "Da_unsaturated.csv")
unsat_da= pd.read_csv(path_da_data)
unsat_data = unsat_da[unsat_da['Chem'].isin (gvarnames)]

#unsat_data["logDa"] = np.log10(unsat_data.Da)

#unsat_data.loc[unsat_data["logDa"] < -1, "PeDamark"] = 0
#unsat_data.loc[(unsat_data["logDa"] > -1) & (unsat_data["logDa"] < 0), "PeDamark"] = 1
#unsat_data.loc[(unsat_data["logDa"] > 0) & (unsat_data["logDa"] <0.5), "PeDamark"] = 2
#unsat_data.loc[(unsat_data["logDa"] > 0.5), "PeDamark"] = 3

#labels = {3 : "log$_{10}$Da > 0.5",
#          2 : "0 < log$_{10}$Da < 0.5",
#          1 : "-1 < log$_{10}$Da < 0",
#         0 : "log$_{10}$Da < -1"}

unsat_data["pc_reldelconc_spatial"] = unsat_data.reldelconc_spatial_fraction * 100

#%%
diff_data_path = os.path.join(uss_dir, "aero_rates_09082022.csv")
mdiff_data = pd.read_csv (diff_data_path)
print (mdiff_data.columns)
print(mdiff_data.rate.unique())
diff_data = mdiff_data[mdiff_data['rate']=='aeration'].reset_index()
diff_data.rename(columns={"rate_val": "aeration"}, inplace=True)
resp_data = mdiff_data[mdiff_data['rate']=='Total_respiration'].reset_index()
resp_data.rename(columns={"rate_val": "Total_resp"}, inplace=True)

#%%
aero_data_path = os.path.join(uss_dir, "massflux_08082022.csv")
mf_data = pd.read_csv (aero_data_path)
print (mf_data.columns)
mf_data['fraction%'] = mf_data.fraction*100
do_data = mf_data[mf_data.Chem=='DO'].reset_index()
do_diff_data = pd.merge(do_data, diff_data[['Regime', 'Trial', 'DOdiffusion']], on = ['Regime', 'Trial'])
aero_data = pd.merge(do_diff_data, resp_data[['Regime', 'Trial', 'Total_resp']], on = ['Regime', 'Trial'])
aero_data["doin_resp_diff"] = -aero_data['massflux_in']+aero_data.DOdiffusion-aero_data.Total_resp
aero_data["resp_diff"] = aero_data.Total_resp/aero_data.DOdiffusion
aero_data["resp_sub_diff"] = (aero_data.Total_resp/(aero_data.DOdiffusion-aero_data.massflux_in))
#%%
fig, a = plt.subplots(3,2, figsize = (8,8), sharey = 'row', sharex = 'col')
a[0,0].text(s="A", x = 0.45, y = 11, **titlekw)
a[0,1].text(s="B", x = 30, y = 11, **titlekw)
a[1,0].text(s="C", x = 0.45, y = 1.1, **titlekw)
a[1,1].text(s="D", x = 30, y = 1.1, **titlekw)
a[2,0].text(s="E", x = 0.45, y = 200, **titlekw)
a[2,1].text(s="F", x = 30, y = 200, **titlekw)
a[2,1].set_xlabel ("Residence time\nof solutes (%)", **titlekw)                 
a[2,0].set_xlabel ("Mean saturation\nin domain", **titlekw)
a[1,0].set_ylabel("Diffusion (uM d-1)", ha='center', va='center', rotation='vertical', labelpad = 15, **titlekw)
a[2,0].set_ylabel(r"$\frac{Respiration}{Diffusion}$", ha='center', va='center', rotation='vertical', labelpad = 15, **titlekw)
#a[0,0].set_xlabel ("")
a[0,0].set_ylabel("DO consumption\nrate (%)", ha='center', va='center', rotation='vertical', labelpad = 15, **titlekw)

sns.scatterplot(x = 'Mean_saturation', y = 'delmassflux', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data, ax = a[0,0])
sns.scatterplot(x = 'fraction%', y = 'delmassflux', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data, ax = a[0,1])
sns.scatterplot(x = 'Mean_saturation', y = 'DOdiffusion', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data, ax = a[1,0])
sns.scatterplot(x = 'fraction%', y = 'DOdiffusion', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data, ax = a[1,1])
sns.scatterplot(x = 'Mean_saturation', y = 'resp_diff', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data, ax = a[2,0])
sns.scatterplot(x = 'fraction%', y = 'resp_diff', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data, ax = a[2,1])
for all in a.flatten():
    all.tick_params(**labelkw)
    all.legend().remove()
for ax in a[2,:]:
    ax.tick_params(**labelkw)
    #ax.set_xscale("log")
    ax.set_yscale("log")
plt.legend(title = "Flow regime", title_fontsize = 14, fontsize = 14, loc = (-0.9,-1), ncol = 3)
picname = os.path.join(op_dir,"Resp_diff_total_Fig_3_diffusion.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)

#%%
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
grey_tri = mlines.Line2D([], [], linestyle = '', marker = "^", markerfacecolor = "grey", markeredgecolor = "grey", markersize=10, label='Nitrate', alpha = 0.5)
grey_dot = mlines.Line2D([], [], linestyle = '', marker = "o", markerfacecolor = "grey", markeredgecolor = "grey", markersize=10, label='DOC', alpha = 0.5)
grey_square = mlines.Line2D([], [], linestyle = '', marker = "s", markerfacecolor = "grey", markeredgecolor = "grey",markersize=10, label='Ammonium', alpha = 0.5)
blue_patch = mpatches.Patch(color="steelblue", label= 'Ratio>10', alpha = 0.5)
orange_patch = mpatches.Patch(color = "orange", label =  '1<Ratio<10', alpha = 0.5)
red_patch = mpatches.Patch(color="green", label= 'Ratio~1', alpha = 0.5)
chem_leg_list = [grey_tri, grey_dot, grey_square]
patchlist = [blue_patch, orange_patch, red_patch]

gvarnames = ["DO", "DOC", "Ammonium", "Nitrate"]
#r_d_ratio = mdiff_data[mdiff_data['rate']=="Respiration_diffusion_ratio_total"].reset_index()
r_d_unsat_data = pd.merge(aero_data[['Regime','Trial', 'Total_resp', 'DOdiffusion','resp_diff']], unsat_data, on = ['Regime','Trial']).reset_index()
#r_d_unsat_data['fraction%'] = r_d_unsat_data.fraction*100
r_d_unsat_data.loc[r_d_unsat_data['resp_diff']>=10, "diff"] = int(0)
r_d_unsat_data.loc[(r_d_unsat_data['resp_diff']>=1) & (r_d_unsat_data['resp_diff']<10), "diff"] = int(1)
r_d_unsat_data.loc[(r_d_unsat_data['resp_diff']<1), "diff"] = int(2)
gvarnames.remove("DO")
#%%
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
        bbox_to_anchor=(0.5, -0.6),
        loc="center",
        title=r"$\frac{Respiration}{Diffusion}$",
        fontsize=14, title_fontsize = 14)
plt.gca().add_artist(legend_flow)
picname = os.path.join(op_dir,"r_d_ratio_Fig_4_Unsaturated_chem_removal.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
#%%
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
            plt.scatter(X*100, y, c = my_pal[d_frac], marker = marklist[frac],alpha = 0.5, label = c)
plt.xlabel ("Residence time of solutes (%)", **titlekw)
plt.ylabel("Removal of reactive species (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
locs, labels1 = plt.yticks()
#plt.yscale("log")
plt.ylim((0,200))
plt.tick_params(**labelkw)
legend_flow = plt.legend(handles=chem_leg_list, ncol = 3,
        bbox_to_anchor=(0.5, -0.5),
        loc="center",
        title="Chemical",
        fontsize=14, title_fontsize = 14)
plt.legend(handles=patchlist, ncol = 3,
        bbox_to_anchor=(0.5, -0.8),
        loc="center",
        title="Respiration vs Diffusion",
        fontsize=14, title_fontsize = 14)
plt.gca().add_artist(legend_flow)
picname = os.path.join(op_dir,"r_d_ratio_Fig_4_2_Unsaturated_chem_removal.png")
#plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
fig, a = plt.subplots(2,2, figsize = (8,8), sharey = 'row', sharex = 'col')
a[0,0].text(s="A", x = 0.45, y = 140, **titlekw)
a[0,1].text(s="B", x = 30, y = 140, **titlekw)
a[1,0].text(s="C", x = 0.45, y = 250, **titlekw)
a[1,1].text(s="D", x = 30, y = 250, **titlekw)
a[1,1].set_xlabel ("Residence time\nof solutes (%)", **titlekw)
a[1,0].set_xlabel ("Mean saturation\nin domain", **titlekw)
a[1,0].set_ylabel(r"$\frac{Respiration}{Diffusion}$", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
a[0,0].set_xlabel ("")
a[0,0].set_ylabel("Impact on microbial activity (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)

sns.scatterplot(x = 'Mean_saturation', y = 'DOrem_fraction%', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data, ax = a[0,0])
sns.scatterplot(x = 'fraction%', y = 'DOrem_fraction%', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = aero_data, ax = a[0,1])
sns.scatterplot(x = 'Mean_saturation', y = 'rate_val', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = diff_data, ax = a[1,0])
sns.scatterplot(x = 'fraction%', y = 'rate_val', hue = 'Regime',
hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = diff_data, ax = a[1,1])
for all in a.flatten():
    all.tick_params(**labelkw)
    all.legend().remove()
for sec_row in a[1,:]:
    sec_row.set_yscale("log")
plt.legend(title = "Flow regime", title_fontsize = 14, fontsize = 14, loc = (-0.9,-0.6), ncol = 3)
picname = os.path.join(op_dir,"Resp_diff_total_Fig_3_diffusion.png")
#plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)

#%%
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
grey_tri = mlines.Line2D([], [], linestyle = '', marker = "^", markerfacecolor = "grey", markeredgecolor = "grey", markersize=10, label='Nitrate', alpha = 0.5)
grey_dot = mlines.Line2D([], [], linestyle = '', marker = "o", markerfacecolor = "grey", markeredgecolor = "grey", markersize=10, label='DOC', alpha = 0.5)
grey_square = mlines.Line2D([], [], linestyle = '', marker = "s", markerfacecolor = "grey", markeredgecolor = "grey",markersize=10, label='Ammonium', alpha = 0.5)
blue_patch = mpatches.Patch(color="steelblue", label= 'Ratio>=5', alpha = 0.5)
orange_patch = mpatches.Patch(color = "orange", label =  '1<=Ratio<5', alpha = 0.5)
red_patch = mpatches.Patch(color="green", label= 'Ratio<1', alpha = 0.5)
chem_leg_list = [grey_tri, grey_dot, grey_square]
patchlist = [blue_patch, orange_patch, red_patch]

gvarnames = ["DO", "DOC", "Ammonium", "Nitrate"]
r_d_ratio = mdiff_data[mdiff_data['rate']=="Respiration_diffusion_ratio_total"].reset_index()
r_d_unsat_data = pd.merge(r_d_ratio[['Regime','Trial','rate', 'rate_val']], unsat_data, on = ['Regime','Trial']).reset_index()
r_d_unsat_data['fraction%'] = r_d_unsat_data.fraction*100
r_d_unsat_data.loc[r_d_unsat_data['rate_val']>=5, "diff"] = int(0)
r_d_unsat_data.loc[(r_d_unsat_data['rate_val']>=1) & (r_d_unsat_data['rate_val']<5), "diff"] = int(1)
r_d_unsat_data.loc[(r_d_unsat_data['rate_val']<1), "diff"] = int(2)
gvarnames.remove("DO")
#%%
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
            plt.scatter(X*100, y, c = my_pal[d_frac], marker = marklist[frac],alpha = 0.5, label = c)
plt.xlabel ("Residence time of solutes (%)", **titlekw)
plt.ylabel("Removal of reactive species (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
locs, labels1 = plt.yticks()
#plt.yscale("log")
plt.ylim((-10,200))
plt.tick_params(**labelkw)
legend_flow = plt.legend(handles=chem_leg_list, ncol = 3,
        bbox_to_anchor=(0.5, -0.35),
        loc="center",
        title="Chemical",
        fontsize=14, title_fontsize = 14)
plt.legend(handles=patchlist, ncol = 3,
        bbox_to_anchor=(0.5, -0.6),
        loc="center",
        title=r"$\frac{Respiration}{Diffusion}$",
        fontsize=14, title_fontsize = 14)
plt.gca().add_artist(legend_flow)
picname = os.path.join(op_dir,"r_d_ratio_Fig_4_2_Unsaturated_chem_removal.png")
#plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#%%
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
            plt.scatter(X*100, y, c = my_pal[d_frac], marker = marklist[frac],alpha = 0.5, label = c)
plt.xlabel ("Residence time of solutes (%)", **titlekw)
plt.ylabel("Removal of reactive species (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
locs, labels1 = plt.yticks()
#plt.yscale("log")
plt.ylim((0,200))
plt.tick_params(**labelkw)
legend_flow = plt.legend(handles=chem_leg_list, ncol = 3,
        bbox_to_anchor=(0.5, -0.5),
        loc="center",
        title="Chemical",
        fontsize=14, title_fontsize = 14)
plt.legend(handles=patchlist, ncol = 3,
        bbox_to_anchor=(0.5, -0.8),
        loc="center",
        title="Respiration vs Diffusion",
        fontsize=14, title_fontsize = 14)
plt.gca().add_artist(legend_flow)
picname = os.path.join(op_dir,"r_d_ratio_Fig_4_2_Unsaturated_chem_removal.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
#%%
subset_numbers = mdiff_data[mdiff_data['rate'].isin(['Respiration_diffusion_>1_num', 'Respiration_diffusion_=1_num', 'Respiration_diffusion_<1_num'])]

for var_to_plot, file_var in zip(["Respiration_diffusion_<1_num", "Respiration_diffusion_>1_num"], ["Resp_diff_g_1_n", "Resp_diff_l_1_n"]):
    subset_numbers["ratio_nodes_fract"] = subset_numbers['rate_val']/6161
    diff_data = subset_numbers[subset_numbers['rate']==var_to_plot]
    diff_data['fraction%'] = diff_data.fraction*100
    fig, a = plt.subplots(1,2, figsize = (8,4), sharey = True)
    sns.scatterplot(x = 'Mean_saturation', y = 'ratio_nodes_fract', hue = 'Regime',
    hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = diff_data, ax = a[0])
    a[0].set_xlabel ("Mean saturation\nin domain", **titlekw)
    a[0].set_ylabel(var_to_plot, ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
    a[0].legend().remove()
    sns.scatterplot(x = 'fraction%', y = 'ratio_nodes_fract', hue = 'Regime',
    hue_order= ['Slow', 'Medium','Fast'], style = 'Regime',palette= my_pal, data = diff_data, ax = a[1])
    #a[0].text(s="A", x = 0.45, y = 8, **titlekw)
    a[1].set_xlabel ("Residence time\nof solutes (%)", **titlekw)
    a[1].set_ylabel ("")
    a[0].tick_params(**labelkw)
    a[1].tick_params(**labelkw)
    #plt.yscale("log")
    #a[1].text(s="B", x = 30, y = 8, **titlekw)
    plt.legend(title = "Flow regime", title_fontsize = 14, fontsize = 14, loc = (-0.9,-0.5), ncol = 3)
    picname = os.path.join(op_dir,file_var+"_Fig_XX_diffusion.png")
    plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.01)



#%%
diff_data_2 = mdiff_data[mdiff_data['rate'].isin(['Respiration_diffusion_>1_num', 'Respiration_diffusion_<1_num'])]
diff_data_2 = diff_data_2.pivot(index = ["Regime", "Trial"], columns = 'rate', values = ['rate_val'])
#diff_unsat_data['ratio_nodes_fract'] = diff_unsat_data.rate_val/6161
diff_data_2['ratio_nodes_fract'] = diff_data_2['rate_val', 'Respiration_diffusion_>1_num']/diff_data_2['rate_val', 'Respiration_diffusion_<1_num']
diff_data_2 = diff_data_2.reset_index()
diff_unsat_data = pd.merge(diff_data_2[['Regime','Trial','ratio_nodes_fract']], unsat_data, on = ['Regime','Trial']).dropna()

#%%
diff_unsat_data.loc[diff_unsat_data[('ratio_nodes_fract', '')]>=1, "diff"] = int(0)
diff_unsat_data.loc[diff_unsat_data[('ratio_nodes_fract', '')]<1, "diff"] = int(1)
diff_labels = {1 : "Diffusion",
         0 : "No diffusion"}
for frac in [1,2,3]:
    subset_1 = diff_unsat_data[diff_unsat_data['PeDamark'] == frac]
    for d_frac in [0,1]:
        subset = subset_1[subset_1['diff']==d_frac]
        if subset.shape[0]<1:
            pass
        else:
            y = subset["pc_reldelconc_spatial"]
            X = subset[["fraction"]]
            plt.scatter(X*100, y, c = my_pal[d_frac], marker = marklist[frac],alpha = 0.5, label = labels[frac])
plt.xlabel ("Residence time of solutes (%)", **titlekw)
plt.ylabel("Removal of reactive species (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
locs, labels1 = plt.yticks()
#plt.yscale("log")
#plt.yticks((0,20,40,60,80,100,120, 140),(0,20,40,60,80,100,120, 140))
#plt.xticks((0,20,40,60,80,100),(0,20,40,60,80,100))
plt.tick_params(**labelkw)
plt.legend(title = "Reactive system", title_fontsize = 14, fontsize = 14, loc = (1.05,0.25))
picname = os.path.join(op_dir,"_Fig_4_Unsaturated_diff_removal_n0_1.png")
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)