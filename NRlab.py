#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Guillermo Palomino Lozano
Organisation: University of Seville
Date: May 2020
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import NRutils as utils

plt.rcParams["font.family"] = "Times New Roman"

def Numerology(Df, FR):
    """
    Description
    ----------
    Specify the numerology according to the standard of 5GNR

    Parameters
    ----------
    Df : float
        subcarrier spacing.
    FR : int
        frequency range.

    Returns
    -------
    MyStructure : dictionary
        information about the numerology specified according to the standard.

    """
    TimeDomainStructure = [{"DF": 15e3,"Tslots":  1e-3, "FR": 1, "Range": [25, 270], "mu": 0},
                           {"DF": 30e3,"Tslots":  0.5e-3, "FR": 1, "Range": [11, 273], "mu": 1},
                           {"DF": 60e3,"Tslots":  0.25e-3, "FR": 1, "Range": [11, 135], "mu": 2},
                           {"DF": 60e3,"Tslots":  0.25e-3, "FR": 2, "Range": [66, 264], "mu": 2},
                           {"DF": 120e3,"Tslots": 0.125e-3, "FR": 2, "Range": [32, 264], "mu": 3},
                           {"DF": 240e3,"Tslots": 0.0625e-3, "FR": 2, "Range": [32, 264], "mu": 4}]
    Dfs = [15e3, 30e3, 60e3, 60e3, 120e3, 240e3]
    #Time-Domain structure definition:
    if Df == 60e3: #According to the standard: 2 possibilities with Df = 60 Khz
        if FR == 1:
            MyStructure = TimeDomainStructure[2]
        else:
            MyStructure = TimeDomainStructure[3]
    else:
        MyStructure = TimeDomainStructure[Dfs.index(Df)]
    return MyStructure

def MulticarrierGeneration(MyStructure, NRB, OvS, Fs, Nslots, M, PAPRd, SEED, PlotOptions, Filtering):

    Ncarr_act = NRB * 12 #Number of active carriers: Each RB means 12 carriers
    FFTSize = int(2**np.ceil(np.log2(Ncarr_act))) #Next power of 2
    if OvS == False:
        OvS = 1
        OvS_freq = 1
        Fs = float(FFTSize * MyStructure["DF"])
        Fo = Fs
    else:
        Fo = float(FFTSize * MyStructure["DF"])
        OvS = Fs / Fo
        if (OvS < 1):
            print("Error. OverSampling lower than 1")
            sys.exit(0)
        OvS_freq = np.floor(OvS)
        OvS_freq = 1
        # Fs is the value given at the input
        print(" Oversampling will be performanced: %.2f" % OvS)
    print(" From Fo = %.2f MHz to Fs = %.2f MHz " % (float(FFTSize * MyStructure["DF"])/1e6, Fs/1e6))
    
    x, Bn, Alphabet, alpha, Eb, const_points, BwCh = tx_OFDM(NRB, Nslots, MyStructure, M, FFTSize, Ncarr_act, Fo, Fs, OvS_freq, SEED, PlotOptions)
    x = x - np.mean(x);
    x  = x  / np.max(np.abs(x));    


    PAPRx = 20*np.log10(np.max(np.abs(x))/np.mean(np.abs(x))) # 10*log10(max(abs(x))/rms(x))
    print(" PAPR before clipping: %f dB" % (PAPRx))
    clip = 10**((PAPRd - PAPRx) / 20)
    idx = np.nonzero(np.abs(x) > clip)
    print(" %d Samples will be affected by clipping" % (np.size(idx)))
    x[idx] = clip*np.exp(1j*np.angle(x[idx]))
    PAPRy = 20*np.log10(np.max(np.abs(x))/np.mean(np.abs(x)))
    print(" PAPR after clipping: %f dB" % (PAPRy))
    if (Filtering == True):
            #do filtering stage here...
            x = utils.Filter_signal(Fs, BwCh*1e6, Ncarr_act*MyStructure["DF"], x)
    if (PlotOptions == True):
        plt.figure()        
        n, bins, patches = plt.hist(x=np.abs(x), bins=20, color='#607c8e',alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Absolute value of the x(n) sample')
        plt.ylabel('Counts')
        plt.title('Histogram of absolute value samples of x(n)')
        # Set a clean upper y-axis limit.
        #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        
        
        t = np.arange(x.size)* 1/Fs
        plt.figure(figsize=(10,4))
        plt.plot(t*10**6,np.abs(x), label='TX signal')
        plt.xlabel('Time [us]'); plt.ylabel('$|x(t)|$');
        plt.grid(True);
        for xc in range(0,Nslots+1):
            plt.axvline(x=(xc * MyStructure["Tslots"] * 1e6),linewidth=2, color='r', label='Slots')
            if xc == 0:
                plt.legend(fontsize=10, loc='lower left') 
        f, Pxx = utils.PSD(x, Fs, FFTSize)
        Pxx  = 10*np.log10(Pxx) + 30
        plt.figure(figsize=(5, 4))
        plt.plot(f,Pxx)
        #plt.plot(f[0:int(FFTSize/2)]/1e6, Pxx_den[0:int(FFTSize/2)],'b', f[int(FFTSize/2):-1]/1e6, Pxx_den[int(FFTSize/2):-1],'b')
        #plt.semilogy(f,np.fft.fftshift(Pxx_den),'b')
        plt.xlabel('frequency [MHz]')
        plt.ylabel('PSD [dBm/Hz]')
        plt.show()
        plt.grid(True);
        plt.title('Power Spectral Density of the signal transmitted')
        sio.savemat('xtime.mat', {'t': t,'x': x})
    print(" Generated OFDM signal has %d samples" % (x.size))
    print(" FFTSize: %d samples" % (FFTSize))
    print("\n------------- FINISHING TX -------------")

    
    #Ncpr = repmat(round(2*OSR*[160; 144*ones(6,1)]*(fs1p4/OSR)/(2*30.72)), Nslots, 1); 1.4MHz
    # fs1p4 = 2*1.92*OSR;
    #Ncpr = repmat(round(2*OSR*[160; 144*ones(6,1)]*(fs15/OSR)/(2*30.72)), Nslots, 1); 15 MHz
    # fs15 = 2*30.72*OSR
    return x, Bn, FFTSize, Alphabet, alpha, Eb, const_points, Fs, Fo, BwCh, PAPRx, PAPRy
    
def tx_OFDM(NRB, Nslots, MyStructure, M, FFTSize, Ncarr_act, Fo, Fs, OvS_freq, SEED, PlotOptions):
    #Range of Channel Bandwidths for the Different Numerologies
    Df = MyStructure["DF"]
    
    if MyStructure["FR"] == 1:
        BwCh = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100])
    else:
        BwCh = np.array([50, 100, 200, 400])
    #Checking that our NRB is in range
    if (NRB < MyStructure["Range"][0]) or (NRB > MyStructure["Range"][1]):
        print("The number of resource blocks does not match the defined numerology\n")
        print("Desired range: %d-%d \n"  % ( MyStructure["Range"][0], MyStructure["Range"][1]))
        print("Your value: %d \n" % (NRB))
        sys.exit(0)
    
    Ncarr_act = NRB * 12 #Number of active carriers: Each RB means 12 carriers
    print(" Number of active subcarriers: %d" % Ncarr_act)
    BwEff = Df * Ncarr_act
    BwEff = (BwEff + 1/2*(BwEff - NRB*12*Df-Df))/1e6 #Eq (18.1)
    #And now we allocate this BW into one of the possibilities defined above
    idxBw = np.nonzero(BwCh > BwEff)[0]
    Bw = BwCh[idxBw[0]] #Now we have set our final Bandwidth
    print(" Channel Bandwidth: %d MHz" % int(Bw))
    print(" Bandwidth occupied: %d MHz" % int(Ncarr_act*Df/1e6))
    NsymbOFDM = Nslots * 14 #14 OFDM symbols every time slot (see Figure 7.2)
    Nres = NsymbOFDM * Ncarr_act #Total resources in terms of symbols
    k = int(np.log2(M)) #Bits per symbol
    Nbits = int(Nres*k) #Total number of bits
    print(" %d bits were generated" % (Nbits))
    np.random.seed(SEED)
    Bn = np.random.randint(0,2,Nbits)
    Eb = 1
    
    #Standard: QPSK, 16QAM, 64QAM and 256QAM
    
    if M != 4 and M != 16 and M != 64 and M!= 256:
        print("Invalid modulation. Formats used within 5G-NR mobile communications system: QPSK, 16QAM, 64QAM and 256QAM.")
        sys.exit(0)
        
    [BnI,BnQ,AnI,AnQ,AI,AQ] = utils.qam(Bn, Eb, np.sqrt(M),np.sqrt(M))
    if (PlotOptions == True):
        plt.figure(figsize=(5,5))
        for ii in range(0,AI.size):
            for jj in range(0,AQ.size):
                plt.plot(AI[ii],AQ[jj], 'ro', label='Constellation')
        plt.xlabel('I'); plt.ylabel('Q');
        plt.grid(True);
        plt.title('Constellation used')
    Alphabet = AI
    const_points = np.zeros((1,AnI.size),dtype=np.complex_)
    const_points.real = AnI
    const_points.imag = AnQ
    const_points = np.reshape(const_points,(Ncarr_act,NsymbOFDM))
    alpha = np.mean(np.abs(const_points))
    
    
    ofdm_freq = np.zeros((FFTSize,NsymbOFDM));
    ofdm_freq = ofdm_freq.astype(np.csingle)
    #Adding information
    aux = 0
    for ii in range(0,NsymbOFDM):
        for jj in range((int)((FFTSize/2)-(Ncarr_act/2)),(int)((FFTSize/2)+(Ncarr_act/2))):
            ofdm_freq[jj,ii] = const_points[aux,ii]
            aux = aux + 1
        aux = 0
    ofdm_freq = utils.OverSampling_Frequency(ofdm_freq,OvS_freq, True)
    if (PlotOptions == True):
        plt.figure()
        plt.stem(np.absolute(ofdm_freq[:,0]), use_line_collection = True)
        plt.xlabel('Subcarriers')
        plt.grid(True)
        plt.title('Subcarriers Tx after Oversampling')
    
    ofdm_freq = np.fft.fftshift(ofdm_freq)    
    ofdm_time = np.zeros(ofdm_freq.shape,dtype=np.complex)   
    for i in range(0,NsymbOFDM):
        ofdm_time[:,i] = np.fft.ifft(ofdm_freq[:,i]) 

    
    def addCP(OFDM_time,NCP):
        cp = OFDM_time[-NCP:]               # take the last CP samples ...
        return np.hstack([cp, OFDM_time])  # ... and add them to the beginning
    
    x = np.array([])
    
    #Cyclic Prefix:
    
    NCP0 = 160/2048*FFTSize *OvS_freq #Overhead - Oversampling done before must be taken into account
    NCP = 144/2048*FFTSize *OvS_freq #Overhead 7%
    
    for ii in range(0,Nslots*2):
        x = np.hstack([x,addCP(ofdm_time[:,ii*7],(int)(NCP0))])
        for jj in range(7*ii+1,7*ii+7):
            x = np.hstack([x,addCP(ofdm_time[:,jj],(int)(NCP))])
    x = utils.FFTinterpolate(x,Fs,Fo)
    
    return x, Bn, Alphabet, alpha, Eb, const_points, Bw

def rx_OFDM(Ncarr_act, NsymbOFDM, y, Alphabet, alpha, FFTSize, OvS_freq, PlotOptions, Simulation, BwCh, Fs):
        Nsamples = int(FFTSize*OvS_freq) #Nsamples -> We have an oversampling that will be fixed in frequency domain
        rx_ofdm_time = np.zeros((Nsamples, NsymbOFDM),dtype=np.complex_)
        y = np.reshape(y,(int(y.size/(int(NsymbOFDM/14*2))),int(NsymbOFDM/14*2)),order='F')
        #Cyclic Prefix:
        NCP0 = int(160/2048*Nsamples) #Overhead 
        NCP = int(144/2048*Nsamples) #Overhead 7
        for ii in range(0,int(NsymbOFDM/14*2)):
            rx_ofdm_time[:,ii*7] = y[NCP0:(NCP0+Nsamples),ii]
            for jj in range(1,7):
                rx_ofdm_time[:,jj+ii*7] = y[NCP0+Nsamples+(NCP+Nsamples)*(jj-1)+NCP:NCP0+Nsamples+(NCP+Nsamples)*(jj),ii]      
        ACPR, ACPR2 = utils.ACPR_Calculation(rx_ofdm_time, Fs, FFTSize, BwCh, OvS_freq)
        print("Adjacent 1st Channel Power Ratio (ACPR): %.2f dB" % ACPR)
        if OvS_freq > 5:
            print("Adjacent 2nd Channel Power Ratio (ACPR): %.2f dB" % ACPR2)
        rx_ofdm_freq = np.zeros((Nsamples, NsymbOFDM),dtype=np.complex_)
        for i in range(0,NsymbOFDM):
            rx_ofdm_freq[:,i] = np.fft.fft(rx_ofdm_time[:,i])
        rx_ofdm_freq = np.fft.fftshift(rx_ofdm_freq)
        if (PlotOptions == True):
            plt.figure()
            plt.stem(np.absolute(rx_ofdm_freq[:,0]), use_line_collection = True)
            plt.xlabel('Subcarriers')
            plt.grid(True)
            plt.title('Subcarriers Rx')
        
        #Downsampling: We want to recover our original signal (Fo MHz instead of Fs MHz)
        rx_ofdm_freq = utils.OverSampling_Frequency(rx_ofdm_freq,OvS_freq,False) #False means we want downsampling (remove zeros)
        rx_const_points = np.zeros((Ncarr_act,NsymbOFDM),dtype=np.complex_)
        aux = 0
        for ii in range(0,NsymbOFDM):
            for jj in range((int)((FFTSize/2)-(Ncarr_act/2)),(int)((FFTSize/2)+(Ncarr_act/2))):
                rx_const_points[aux,ii] = rx_ofdm_freq[jj,ii]
                aux = aux + 1
            aux = 0
        # From subcarriers to symbols...
        beta = np.mean(np.abs(rx_const_points))
        rx_const_points = rx_const_points * (alpha/beta)
        rx_AnI = rx_const_points.real
        rx_AnQ = rx_const_points.imag
        rx_AnI = rx_AnI.flatten()
        rx_AnQ = rx_AnQ.flatten()
        est_AnI = utils.detectsymb(rx_AnI, Alphabet)
        est_AnQ = utils.detectsymb(rx_AnQ, Alphabet)
        rx_BnI = utils.symb2bit(est_AnI, Alphabet)
        rx_BnQ = utils.symb2bit(est_AnQ, Alphabet)
        k = int(np.log2(Alphabet.size))
        rx_Bn = np.array([],dtype=np.int)
        # ... and now bits (parallel to series conversion included here)
        for ii in range(0,int(rx_BnI.size/k)):
            rx_Bn = np.hstack([rx_Bn,rx_BnI[ii*k:ii*k+k]])
            rx_Bn = np.hstack([rx_Bn,rx_BnQ[ii*k:ii*k+k]])
        return rx_Bn, rx_const_points, ACPR, ACPR2
    
def Channel(x, SNRdb, Chtype):
    """
    Description
    ----------
    Channel model of the simulated environment

    Parameters
    ----------
    x : numpy of array complex
        input signal.
    SNRdb : float
        signal to noise ratio specified in decibels.
    Chtype : string
        Channel type.

    Returns
    -------
    y : numpy of array complex
        output signal of the channel stage.

    """
    if (Chtype == "AWGN"):
        nsr = 10**(-SNRdb/10)
        Ps = np.sum(np.abs(x**2))/x.size
        n = np.random.randn(len(x)*2).view(np.complex128)/np.sqrt(2) #Noise with power 1
        n = np.sqrt(Ps*nsr)*n #We set noise power to meet SNR specifications
        y = x + n
    return y

def MulticarrierReceiver(NRB, MyStructure, y, Nslots, Alphabet, alpha, FFTSize, Fs, Fo, PlotOptions, Simulation, BwCh, Filtering):
    Ncarr_act = NRB * 12
    NsymbOFDM = Nslots*14
    OvS = Fs / Fo
    OvS_freq = np.floor(OvS)
    f, Pyy = utils.PSD(y, Fs, FFTSize)
    Pyy  = 10*np.log10(Pyy) + 30
    if (PlotOptions == True):
        plt.figure(figsize=(5, 4))
        plt.plot(f,Pyy)
        plt.xlabel('frequency [MHz]')
        plt.ylabel('PSD [dBm/Hz]')
        plt.title('Power Spectral Density of the signal received')
        plt.show()
        plt.grid(True);
    
    #Interpolation in time domain (Downsampling)
    y = utils.FFTinterpolate(y,Fo*OvS_freq,Fs)
    
    rx_Bn, rx_const_points, ACPR, ACPR2 = rx_OFDM(int(Ncarr_act), int(NsymbOFDM), y, Alphabet, alpha,FFTSize, OvS_freq, PlotOptions, Simulation, BwCh, Fs)
    return rx_Bn, rx_const_points, ACPR, ACPR2

