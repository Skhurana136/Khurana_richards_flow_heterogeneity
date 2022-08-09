#%%
# ## Import libraries
import os
import pandas as pd
import numpy as np

#User libraries
#import DS.plots.saturated_steady_state as sssp
#import DS.analyses.transient as translys
import DS.data_reader.data_processing as proc
print ("All libraries loaded")

# Assign directories based on where you are
# work computer
data_dir = r"C:\Users\swkh9804\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
raw_dir = r"D:\Data\Richards_flow\RF_big_sat_2"
# home computer
#data_dir = r"C:\Users\swami\OneDrive\Documents\Manuscripts\Paper3\Figurecodes"
#raw_dir = r"E:\Data\Richards_flow\Richards_flow_big_sat"#D:\Data\Richards_flow\RF_big_sat_2"
# %%
#Assign file names
biomass_file = "biomass_comparison_09082022.csv"
chem_file = "conc_comparison_09082022.csv"
print ("Path directories and files assigned")

filename = "concdata_09082022.csv"
mfdata = pd.read_csv(os.path.join(data_dir, filename))
print(mfdata.shape)
print(mfdata.columns)
filename = "conc_comparison_09082022.csv"
mfdata_comparison = pd.read_csv(os.path.join(data_dir, filename))
print(mfdata_comparison.shape)
print(mfdata_comparison.columns)

# How many regimes and what are they called? Which chemical species?
print(mfdata.Regime.unique())
print(mfdata.Chem.unique())
print(mfdata_comparison.Regime.unique())
print(mfdata_comparison.Chem.unique())

def Convertetoarray(D, datatype, vertnodes, horiznodes):
    if (datatype == "tec"):
        steps = np.shape(D)[2]
        Headers = np.shape(D)[1]
        size = np.shape(D)[0]
        df2 = np.ndarray([Headers - 3, steps, vertnodes, horiznodes])
        for i in range(Headers - 3):
            counter = 0
            for j in range(steps):
                #           print (Headers[i+3])
                while counter < (j + 1) * size:
                    for k in range(vertnodes):
                        for l in range(horiznodes):
                            df2[i, j, k, l] = D[counter - (j) * size, i + 3, j]
                            counter = counter + 1
        df = np.ndarray([Headers - 3, steps, vertnodes, horiznodes])
        for j in range(vertnodes):
            df[:, :, vertnodes - 1 - j, :] = df2[:, :, j, :]
    elif (datatype == "rates"):
        Headers = np.shape(D)[1]
        size = np.shape(D)[0]
        df2 = np.ndarray([Headers, vertnodes, horiznodes])
        for i in range(Headers):
            counter = 0
            while counter < size:
                for k in range(vertnodes):
                    for l in range(horiznodes):
                        df2[i, k, l] = D[counter - size, i]
                        counter = counter + 1
        df = np.ndarray([Headers, vertnodes, horiznodes])
        for j in range(vertnodes):
            df[:, vertnodes - 1 - j, :] = df2[:, j, :]        
    return df

allrates = proc.masterrates("Unsaturated")
allrates
yin = 6
yout = -7
aerorates = ['Fixedaeroresp', 'Mobaeroresp','Fixedammresp','Mobammresp','DOdiffusion']
ratesind = list(19+allrates.index(a) for a in aerorates)

Regimes = ["Medium","Slow","Fast"]
trialist = proc.masterscenarios("Unsaturated")
mTrial = list(trialist.keys())
droplist = ["72","98","111","137","165","185"]
Trial = list(t for t in mTrial if t not in droplist)
#%%
row = []
for r in Regimes:
    reg_dir = os.path.join(raw_dir, r+"AR_0")
    for t in Trial:
        print(r, t)
        path = os.path.join(reg_dir, "RF-A"+str(t),"ratesAtFinish.dat")
        rates = np.loadtxt(path, usecols = ratesind, delimiter = ' ')
        data_path = os.path.join(raw_dir, r + "AR_0" + "_RF-A"+str(t) + "_df.npy")
        data = np.load(data_path)
        sat = data[4,-1,:,:]
        rates_array = Convertetoarray(rates, "rates", 113, 61)
        resp_sum = 0
        resp_corners = 0
        resp_boundary_top = 0
        resp_boundary_side = 0
        resp_elem = 0
        diff_i = aerorates.index('DOdiffusion')
        diff_corners = [rates_array[diff_i,yin,0],rates_array[diff_i,yin,-1],rates_array[diff_i,yout,0],rates_array[diff_i,yout,-1]]
        diff_boundary_top = [rates_array[diff_i,yin,1:-1],rates_array[diff_i,yout,1:-1]]
        diff_boundary_side = [rates_array[diff_i,yin+1:yout,0],rates_array[diff_i,yin+1:yout,-1]]
        diff_elem = rates_array[diff_i,yin+1:yout,1:-1]
        sat_corners = np.asarray([sat[yin,0],sat[yin,-1],sat[yout,0],sat[yout,-1]])
        sat_boundary_top = np.asarray([sat[yin,1:-1],sat[yout,1:-1]])
        sat_boundary_side = np.asarray([sat[yin+1:yout,0],sat[yin+1:yout,-1]])
        sat_elem = sat[yin+1:yout,1:-1]
        diff_total = np.sum(diff_corners*sat_corners)*0.0025*0.0025+(np.sum(diff_boundary_top*sat_boundary_top)+np.sum(diff_boundary_side*sat_boundary_side))*0.0025*0.005+np.sum(diff_elem*sat_elem)*0.005*0.005
        row.append([r,t,"DO",'aeration',diff_total])
        for r_i in list(range(len(aerorates)-1)):
            corners = np.asarray([rates_array[r_i,yin,0],rates_array[r_i,yin,-1],rates_array[r_i,yout,0],rates_array[r_i,yout,-1]])
            boundary_top = np.asarray([rates_array[r_i,yin,1:-1],rates_array[r_i,yout,1:-1]])
            boundary_side = np.asarray([rates_array[r_i,yin+1:yout,0],rates_array[r_i,yin+1:yout,-1]])
            elem = rates_array[r_i,yin+1:yout,1:-1]
            totaldorem = np.sum(corners*sat_corners)*0.0025*0.0025+(np.sum(boundary_top*sat_boundary_top)+np.sum(boundary_side*sat_boundary_side))*0.0025*0.005+np.sum(elem*sat_elem)*0.005*0.005
            resp_corners = resp_corners + corners
            resp_boundary_top = resp_boundary_top + boundary_top
            resp_boundary_side = resp_boundary_side +  boundary_side
            resp_elem += elem
            resp_sum += totaldorem
            row.append([r,t,"DO",aerorates[r_i],totaldorem])
        row.append([r,t,"DO","Total_respiration",resp_sum])
netdorem = pd.DataFrame.from_records(row, columns = ["Regime", "Trial", "Chem","rate","rate_val"])

sat_dorem = pd.merge(mfdata_comparison, netdorem, on = ["Regime", "Trial", "Chem"])

all_rates = list(sat_dorem.rate.unique())
for r in Regimes:
    for t in Trial:
        for c in all_rates:
            if c=='DOdiffusion':
                c = "aeration"
            spat_n_base = sat_dorem.loc[(sat_dorem.Regime == r) & (sat_dorem.rate == c) & (sat_dorem.Trial == 'H')]['rate_val'].values[0]
            sat_dorem.loc[(sat_dorem.Regime == r) & (sat_dorem.rate == c) & (sat_dorem.Trial == t), 'spatial_base'] = spat_n_base

sat_dorem['rate_spatial_fraction'] = sat_dorem['rate_val']/sat_dorem['spatial_base']

sat_dorem.to_csv(os.path.join(data_dir,"aero_rates_09082022.csv"), index =False)