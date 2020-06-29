#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Guillermo Palomino Lozano
Organisation: University of Seville
Date: May 2020
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio
#import matlab.engine
from matplotlib.ticker import ScalarFormatter
import warnings

import NRlab as NR
import NRutils as utils


#------------------------------------------------------------
plt.close('all')

NRB = 150 # Number of resource blocks
Df = 30e3 # Subcarrier Spacing
OvS = True # Oversampling
Fs = 200e6 # Only important when oversampling is true
Nslots = 1
FR = 1 # Frecuency range
M = 64 #Modulation order
PAPR = 10.5 #PAPR desired
SEED = 1000 # Seed for random number generation
PlotOptions = True  #True for active plots (Subcarriers Tx/Rx, Constellation, Spectrum, etc )
#-----------------------------------------------Pl-------------
#   Simulations available:
BERsimulation = False
Amplifiersimulation = False
RunMatlabFunction = False
Filtering = True

#------------------------------------------------------------
warnings.filterwarnings('ignore')
if (BERsimulation == True):
    #Modulation = np.array([4])
    Modulation = np.array([4, 16, 64, 256])  # Modulation order
    EbNo = np.arange(0,15,1)
    BER = np.zeros((len(EbNo),len(Modulation)))
    errorteorico = np.zeros((len(EbNo),len(Modulation)))
    EVM = np.zeros((len(EbNo),len(Modulation)))
    NMSE = np.zeros((len(EbNo),len(Modulation)))
else:
    Modulation = np.array([M])  # Modulation order
    EbNo = np.array([float('inf')]) 
    BER = 0
for jj in range(len(Modulation)):
    M = Modulation[jj]
    k = np.log2(M)
    MyStructure = NR.Numerology(Df,FR)
    print("\n------------- STARTING TX -------------")
    print("\n M = %d" %(M))
    x, Bn, FFTSize, Alphabet, alpha, Eb, const_points, Fs, Fo, BwCh, PAPRx, PAPRy = NR.MulticarrierGeneration(MyStructure, NRB, OvS, Fs, Nslots, M, PAPR, SEED, PlotOptions, Filtering)
    #Ready for real measurements!
    if (RunMatlabFunction == True):
        #RunBERsimulation.m implementation goes here
        Nmeas = 3
        Pin = np.arange(-30,-20)
        #Pin = np.array([-30,-20])
        X_amp = np.zeros((Pin.size,x.size),dtype=np.complex128)
        Y_amp = np.zeros((Pin.size,x.size),dtype=np.complex128)
        currpath = os.getcwd()
        os.chdir("Chalmers") 
        print("\n------------- STARTING Power Amplifier stage -------------")
        print("Starting Matlab engine...")
        eng = matlab.engine.start_matlab()
        for ii in range(len(Pin)):
            RMSin = Pin[ii]
            print("\n Saving signal for real measurements...")
            sio.savemat('NR_signal.mat', {'x':x,'RMSin':RMSin,'Nmeas':Nmeas})
            print(" Running real measurement in the RF WebLab for Pin = %d dBm ... it will take around %d minute(s) ..." % (int(RMSin), int(Nmeas)))
            #os.system("NRChalmers.exe")
            eng.NRChalmers(nargout=0)
            print(" done!")
            mat = sio.loadmat('NR_signal_Output.mat')
            aux = mat["x"][:,0]
            X_amp[ii,:].real = mat["x"][:,0].real
            X_amp[ii,:].imag = mat["x"][:,0].imag
            Y_amp[ii,:].real = mat["y"][:,0].real
            Y_amp[ii,:].imag = mat["y"][:,0].imag
            #data = mat["data"]
        #Save the results in a file
        eng.quit()
        os.chdir("../")
        np.savez('PowAmp.npz',X_amp=X_amp,Y_amp=Y_amp,Pin=Pin) 
        y_amp = Y_amp[1,:]
        print("\n------------- FINISHING Power Amplifier stage -------------")
    if (BERsimulation == False):
        print("\nGenerated signal:")
        print(x)
        print("\nBits transmitted:")
        print(Bn)
    print("\n------------- STARTING RX -------------")
    for ii in range(len(EbNo)):
        if (Amplifiersimulation == True):
            data = np.load('PowAmp.npz')
            Xamp = data['X_amp']
            Yamp = data['Y_amp']
            Pin = data['Pin']
            Nsim, Nsamples = Yamp.shape
            for jj in range(Nsim):
                print("\n Analyzing measurement for Pin = %.2f dBm" % Pin[jj])
                y = NR.Channel(Yamp[jj,:], EbNo[ii], Chtype="AWGN")
                rx_Bn, rx_const_points, ACPR, ACPR2 = NR.MulticarrierReceiver(NRB, MyStructure, y, int(Nslots), Alphabet, alpha, int(FFTSize), Fs, Fo, PlotOptions, BERsimulation,BwCh, Filtering)
                utils.DrawConstellation(const_points.flatten(), rx_const_points.flatten(), NUMPOINTS = 500)
        else:
            SNRdB=EbNo[ii]+10*np.log10(k)+10*np.log10(NRB*12/FFTSize)
            y = NR.Channel(x, SNRdB, Chtype="AWGN")
            if (BERsimulation == False):
                print("Received signal:")
                print(x)
            rx_Bn, rx_const_points, ACPR, ACPR2 = NR.MulticarrierReceiver(NRB, MyStructure, y, int(Nslots), Alphabet, alpha, int(FFTSize), Fs, Fo, PlotOptions, BERsimulation,BwCh, Filtering)
            if (PlotOptions == True):
                utils.DrawConstellation(const_points.flatten(), rx_const_points.flatten(),NUMPOINTS = 500)  
            if (BERsimulation == False):
                print("Received sequence:")
                print(rx_Bn)
            
            if (BERsimulation == True):
                print(" EbNo = %d dB" % (EbNo[ii]))
                BER[ii,jj] = np.sum((Bn != rx_Bn))/Bn.size
                #print("\nBER para SNR = %f dB: %f" % (EbNo[ii], BER[ii,jj]))
                ebno = 10**(EbNo[ii]/10)
                errorteorico[ii,jj] = 4/k*utils.Qfunct(np.sqrt(3*ebno*k/(M-1)))
                #arg = np.sqrt(3*k/(M-1)/No)
                #errorteorico[ii,jj] = (4/k)*(1-1/np.sqrt(M))*Qfunct(arg)
                EVM[ii,jj] = utils.EVM_Calculation(const_points.flatten(), rx_const_points.flatten())
                NMSE[ii,jj] = utils.NMSE_Calculation(x,y)
    print("\n------------- FINISHING RX -------------")
if (BERsimulation == True):
    plt.figure()
    plt.semilogy(EbNo, errorteorico[:,0], '-',label='M = 4 (Th)', color='blue')
    plt.semilogy(EbNo, errorteorico[:,1], '-',label='M = 16 (Th)', color='red')
    plt.semilogy(EbNo, errorteorico[:,2], '-',label='M = 64 (Th)', color='green')
    plt.semilogy(EbNo, errorteorico[:,3], '-',label='M = 256 (Th)', color='black')
    plt.semilogy(EbNo, BER[:,0],'-o', label='M = 4', color='blue')
    plt.semilogy(EbNo, BER[:,1],'-o', label='M = 16', color='red')
    plt.semilogy(EbNo, BER[:,2],'-o', label='M = 64', color='green')
    plt.semilogy(EbNo, BER[:,3],'-o', label='M = 256', color='black')
    plt.xlabel('EbNo [dB]'); plt.ylabel('BER');
    plt.grid(True,which="both",ls="-")
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([10**(-7),0.5])
    plt.title('5G NR System performance for AWGN channel')
    plt.savefig('../results/berperformance.png')
    
    plt.figure()
    fig, ax = plt.subplots()
    for plt.axis in [ax.xaxis, ax.yaxis]:
        plt.axis.set_major_formatter(ScalarFormatter())
    plt.semilogy(EbNo, EVM[:,0],'-o',label='M = 4', color='blue')
    plt.semilogy(EbNo, EVM[:,1],'-o',label='M = 16', color='red')
    plt.semilogy(EbNo, EVM[:,2],'-o',label='M = 64', color='green')
    plt.semilogy(EbNo, EVM[:,3],'-o',label='M = 256', color='black')
    plt.xlabel('EbNo [dB]'); plt.ylabel('EVM (%)');
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.title('5G NR System performance for AWGN channel')
    plt.savefig('../results/evmperformance.png')
    
    plt.figure()
    fig, ax = plt.subplots()
    for plt.axis in [ax.xaxis, ax.yaxis]:
        plt.axis.set_major_formatter(ScalarFormatter())
    plt.semilogy(EbNo, NMSE[:,0],'-o',label='M = 4', color='blue')
    plt.semilogy(EbNo, NMSE[:,1],'-o',label='M = 16', color='red')
    plt.semilogy(EbNo, NMSE[:,2],'-o',label='M = 64', color='green')
    plt.semilogy(EbNo, NMSE[:,3],'-o',label='M = 256', color='black')
    plt.xlabel('EbNo [dB]'); plt.ylabel('NMSE');
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.title('5G NR System performance for AWGN channel')
    plt.savefig('../results/nmseperformance.png')
else:
    EVM = utils.EVM_Calculation(const_points.flatten(), rx_const_points.flatten())
    print("EVM: %.2f" % EVM)