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

#set up basic constants 
Regimes = ["Slow", "Equal", "Fast"]
domains = ["", "Half", "Double", "Big"]
ynodes = 51
horiznodes = 31
scdict = proc.masterscenarios() #master dictionary of all spatially heterogeneous scenarios that were run
filename = "model_domain_quad.tec" #same filename in each subfolder
Trial = list(t for t,values in scdict.items())

# Reading and storing in numpy array
for Regime in Regimes:
    readdirectory = "X:/Richards_flow/" + Regime + "AR_0/RF-A" #change directory as per flow regime
    writedirectory = "E:/Richards_flow/" + Regime + "AR_0/" #change directory as per flow regime
    for j in Trial:
        print(j)
        fwithd = os.path.join(readdirectory + str(j), filename) #complete path to file
        print("Reading tech file....")
        size, steps, Headers, D = rdr.readTecfile(fwithd) #read the tec file
        print("Converting to array....")
        df = rdr.Convertetoarray(D, "tec", ynodes, horiznodes) #Convert the coarse grid in to 51x31 array
        print("Saving numpy array....") 
        np.save(os.path.join(writedirectory, Regime + "AR_0_RF-A"+j +  "_df"), df)
        # Test for correct orientation of the data
        for i in range(np.shape(df)[0]):
            print(Headers[i + 3], np.mean(df[i, -1, 0, :]))
        print(np.mean(df[2, -1, :, :]))