## Import libraries
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
raw_dir = r"D:\Data\Richards_flow\Richards_flow_big_sat"#D:\Data\Richards_flow\RF_big_sat_2"
# %%
#Assign file names
biomass_file = "biomass_comparison_with_sat_26092021.csv"
chem_file = "conc_comparison_with_sat_26092021.csv"
print ("Path directories and files assigned")

filename = "concdata_with_sat_26092021.csv"#concdata_03082021.csv"
mfdata = pd.read_csv(os.path.join(data_dir, filename))
print(mfdata.shape)
print(mfdata.columns)
filename = "conc_comparison_with_sat_26092021.csv"#03082021.csv"
mfdata_comparison = pd.read_csv(os.path.join(data_dir, filename))
#filename = "E:/Richards_flow/RF_big_sat_2/Da_unsaturated.csv"
#mfdata_comparison = pd.read_csv(filename)
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

aerorates = ['Fixedaeroresp', 'Mobaeroresp','Fixedammresp','Mobammresp','DOdiffusion']
ratesind = list(19+allrates.index(a) for a in aerorates)

Regimes = ["Medium","Slow","Fast"]
trialist = proc.masterscenarios("Unsaturated")
mTrial = list(trialist.keys())
droplist = ["72","98","111","137","165","185"]
Trial = list(t for t in mTrial if t not in droplist)

row = []
for r in Regimes:
    reg_dir = os.path.join(raw_dir, r+"AR_0")
    for t in Trial:
        path = os.path.join(reg_dir, "RF-A"+str(t),"ratesAtFinish.dat")
        print(r,t)
        rates = np.loadtxt(path, usecols = ratesind, delimiter = ' ')
        rates_array = Convertetoarray(rates, "rates", 113, 61)
        resp_sum = 0
        for r_i in list(range(len(aerorates))):
            corners = (rates_array[r_i,6,0]+rates_array[r_i,6,-1]+rates_array[r_i,-7,0]+rates_array[r_i,-7,-1])*0.0025*0.0025
            boundary = np.sum(np.sum(rates_array[r_i,6,1:-1]+rates_array[r_i,-7,1:-1])*0.0025*0.005+(rates_array[r_i,6:-6,0]-rates_array[r_i,6:-6,-1])*0.0025*0.005)
            elem = np.sum(rates_array[r_i,6:-6,1:-1])*0.005*0.005
            netdorem = corners+boundary+elem
            resp_sum += netdorem
            row.append([r,t,"DO",aerorates[r_i],netdorem])
        resp_diff_ratio = (resp_sum-netdorem)/netdorem
        row.append([r,t,"DO","Respiration_diffusion_ratio",resp_diff_ratio])

netdorem = pd.DataFrame.from_records(row, columns = ["Regime", "Trial", "Chem","rate","rate_val"])

sat_dorem = pd.merge(mfdata_comparison, netdorem, on = ["Regime", "Trial", "Chem"])

sat_dorem.head()

#for r in Regimes:
#    base = sat_dorem[(sat_dorem['Regime']==r) & (sat_dorem['Trial']=="H")]["diffusion"].values[0]
#    for t in Trial:
#        sat_dorem.loc[(sat_dorem['Regime']==r) & (sat_dorem['Trial']==t),"diffusion_base"]=base
#sat_dorem['diffusion_fraction'] = sat_dorem['diffusion']/sat_dorem['diffusion_base']
#sat_dorem['diffusion_fraction%'] = sat_dorem['diffusion_fraction']*100

sat_dorem.to_csv(os.path.join(data_dir,"aero_rates_het.csv"), index =False)