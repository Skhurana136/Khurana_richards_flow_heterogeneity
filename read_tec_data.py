# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:42:48 2020

@author: khurana
"""
#Reading tec files and storing them as numpy array - possible options to read 51x31 matrices or 101x61 matrices in data_reader library
import os
import numpy as np
import data_reader.reader as rdr
import data_reader.data_processing as proc

#directories
parent_dir = "Z:/Richards_flow_tr"#big_sat"#"X:/Richards_flow_bot_sat"
#set up basic constants 
Regimes = ["Slow","Fast","Medium"]
ynodes = 113
horiznodes = 61

scdict = proc.masterscenarios("Unsaturated") #master dictionary of all spatially heterogeneous scenarios that were run
filename = "model_domain_quad.tec" #same filename in each subfolder
Trial = list(t for t,values in scdict.items())

droplist = []
# Reading and storing in numpy array
for Regime in Regimes:
    regdirectory =  os.path.join(parent_dir, Regime + "AR_0") #change directory as per flow regime
    for j in Trial:
        if j in droplist:
            pass
        else:
            print(j)
            try:
                fwithd = os.path.join(regdirectory, "RF-A" + j, filename) #complete path to file
                print("Reading tech file....")
                size, steps, Headers, D = rdr.readTecfile(fwithd) #read the tec file
                print("Converting to array....")
                df = rdr.Convertetoarray(D, "tec", ynodes, horiznodes) #Convert the coarse grid in to 51x31 array
                print("Saving numpy array....") 
                np.save(os.path.join(regdirectory, Regime + "AR_0_RF-A"+j +  "_df"), df)
                # Test for correct orientation of the data
                for i in range(np.shape(df)[0]):
                    print(Headers[i + 3], np.mean(df[i, -1, 0, :]))
                print(np.mean(df[2, -1, :, :]))
            except:
                print(Regime + " " + j + " failed")
                continue

#directories
parent_dir = "X:/Richards_flow_bot_sat"
#set up basic constants 
Regimes = ["Slow"]
ynodes = 113
horiznodes = 61
filename = "model_domain_RICHARDS_FLOW_quad.tec" #same filename in each subfolder
filename = "model_domain_quad.tec"
Trialr = list(range(175,185,1))
Trial = list(str(t) for t in Trialr)

droplist = []
# Reading and storing in numpy array
for Regime in Regimes:
    regdirectory =  os.path.join(parent_dir, Regime + "AR_0") #change directory as per flow regime
    for j in Trial:
        if (Regime == "Slow") and (j in droplist):
            pass
        else:
            print(j)
            try:
                fwithd = os.path.join(regdirectory, "RF-A" + j, filename) #complete path to file
                print("Reading tech file....")
                size, steps, Headers, D = rdr.readTecfile(fwithd) #read the tec file
                print("Converting to array....")
                df = rdr.Convertetoarray(D, "tec", ynodes, horiznodes) #Convert the coarse grid in to 51x31 array
                print("Saving numpy array....") 
                np.save(os.path.join(regdirectory, Regime + "AR_0_RF-A"+j +  "_df"), df)
                # Test for correct orientation of the data
                for i in range(np.shape(df)[0]):
                    print(Headers[i + 3], np.mean(df[i, -1, 0, :]))
                print(np.mean(df[2, -1, :, :]))
            except:
                print(Regime + " " + j + " failed")
                continue
velindex = 2
vedge = 0.01/2
velem = 0.01
por = 0.2
yin = [4,5,6]
yout = [-6,-5,-4]
voly=0.005 #0.01
def effsat(data):
    slope = 1/(0.8-0.2)
    constant = -1/3
    sat = slope*data + constant
    return sat
yin = [6]
yout = [-6]
for Regime in Regimes:
    regdirectory =  os.path.join(parent_dir, Regime + "AR_0") #change directory as per flow regime
    for j in Trial:
        df = np.load(os.path.join(regdirectory, Regime + "AR_0_RF-A"+j +  "_df.npy"))
        for yi in yin:
            for yo in yout:
                velx = df[1, -1, :, :]
                vely = df[2, -1, :, :]
                sat = df[4,-1,:,:]
                veledgi = (vely[yi,0]*sat[yi,0] + vely[yi,-1]*sat[yi,-1])*0.005*voly
                veledgo = (vely[yo,-1]*sat[yo,-1] + vely[yo,-1]*sat[yo,-1])*0.005*voly
                sumvelin = sum(vely[yi,1:-1]*sat[yi,1:-1]+vely[yi,1:-1]*sat[yi,1:-1])*voly*0.01 + veledgi
                sumvelout = sum(vely[yo,1:-1]*sat[yo,1:-1]+vely[yo,1:-1]*sat[yo,1:-1])*voly*0.01 + veledgo
                #sumvelin = sum(vely[yi,:]*sat[yi,:])
                #sumvelout = sum(vely[yo,:]*sat[yo,:])
                delta = sumvelin-sumvelout
                print(j, yi, yo, delta/sumvelin)

#parent_dir = "C:/Users/khurana/Documents/Scripts/ogs_plain"
#filename = "model_domain_RICHARDS_FLOW_quad.tec" #same filename in each subfolder
parent_dir = "X:/Richards_flow_bot_sat/SlowAR_0/RF-A189"
filename = "model_domain_quad.tec" #same filename in each subfolder
fwithd = os.path.join(parent_dir, filename) #complete path to file
print("Reading tech file....")
size, steps, Headers, D = rdr.readTecfile(fwithd) #read the tec file
print("Converting to array....")
df = rdr.Convertetoarray(D, "tec", ynodes, horiznodes) #Convert the coarse grid in to 51x31 array
print("Saving numpy array....") 
np.save(os.path.join(parent_dir, "df"), df)
# Test for correct orientation of the data
#for i in range(np.shape(df)[0]):
#    print(Headers[i + 3], np.mean(df[i, -1, 0, :]))
print(np.mean(df[2, -1, :, :]))
gvarnames = ["P","VX", "VY", "Sat", "Tracer"]
gindx = [0,1,2,4]
velindex = 2
vedge = 0.01/2
velem = 0.01
por = 0.2
yin = [6]
yout = [-6]
voly=0.005 #0.01
velh = 2.34e-06
def effsat(data):
    slope = 1/(0.8-0.2)
    constant = -1/3
    sat = slope*data + constant
    return sat

for yi in yin:
    for yo in yout:
        velx = df[1, -1, :, :]
        vely = df[2, -1, :, :]
        sat = df[4,-1,:,:]
        veledgi = (vely[yi,0]*sat[yi,0] + vely[yi,-1]*sat[yi,-1])*0.005*voly
        veledgo = (vely[yo,-1]*sat[yo,-1] + vely[yo,-1]*sat[yo,-1])*0.005*voly
        sumvelin = sum(vely[yi,1:-1]*sat[yi,1:-1]+vely[yi,1:-1]*sat[yi,1:-1])*voly*0.01 + veledgi
        sumvelout = sum(vely[yo,1:-1]*sat[yo,1:-1]+vely[yo,1:-1]*sat[yo,1:-1])*voly*0.01 + veledgo
        #sumvelin = sum(vely[yi,:]*sat[yi,:])
        #sumvelout = sum(vely[yo,:]*sat[yo,:])
        delta = sumvelin-sumvelout
        diff = sumvelin + velh
        print(yi, yo, delta/sumvelin, diff/velh)
