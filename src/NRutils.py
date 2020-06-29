#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Guillermo Palomino Lozano
Organisation: University of Seville
Date: May 2020
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.signal import kaiserord, firwin, freqz, welch
from scipy.special import erfc
from sympy.combinatorics.graycode import GrayCode


def PSD(y, Fs, FFTSize):
    """
    Description
    -----------
    Compute the power spectral density for a given signal with a frequency sampling Fs and using FFTSize points
    
    Parameters
    ----------
    y : numpy complex array
        input signal.
    Fs : float
        frecuency sampling.
    FFTSize : int
        number of points required for the computing of the fast fourier transform algorithm.

    Returns
    -------
    f : numpy float array
        frequency grid.
    Pxx : numpy complex array
        power spectral density of the given signal y.

    """
    
    f_aux, Pxx_aux = welch(y, Fs, nperseg=FFTSize)
    f = np.array([])
    Pxx = np.array([])
    length = len(f_aux)
    f=np.concatenate((f,f_aux[int(FFTSize/2):length]))
    f=np.concatenate((f,f_aux[0:int(FFTSize/2)]))
    f = f/1e6
    Pxx=np.concatenate((Pxx,Pxx_aux[int(FFTSize/2):length]))
    Pxx=np.concatenate((Pxx,Pxx_aux[0:int(FFTSize/2)]))
    return f, Pxx
    
def ACPR_Calculation(y, Fs, FFTSize, BwCh, OvS):
    """
    Description
    ----------
    Compute the adjacent channel power ratio (ACPR) for a given signal y

    Parameters
    ----------
    y : numpy complex array
        input signal.
    Fs : float
        frecuency sampling.
    FFTSize : int
        number of points required for the computing of the fast fourier transform algorithm.
    BwCh : int
        Channel bandwidth.
    OvS : float
        Oversampling needed for meet the requirement of the frequency sampling desired.

    Returns
    -------
    ACPR : float
        ACPR of the first adjacent channel.
    ACPR2 : float
        ACPR of the second adjacent channel.

    """
    
    Nsamples, Nsymb = y.shape
    Pchann = np.zeros(Nsymb)
    Pabov = np.zeros(Nsymb)
    Pbelw = np.zeros(Nsymb)

    Pabov2 = np.zeros(Nsymb)
    Pbelw2 = np.zeros(Nsymb)
    for ii in range(Nsymb):
        yaux = y[:,ii]
        
        f, psdy = PSD(yaux, Fs, FFTSize)
        
        idx_chann = (f > -(BwCh/2)) * (f < (BwCh/2))
        idx_abov = (f > (BwCh/2)) * (f < ((BwCh/2) + BwCh))
        idx_belw = (f < (-BwCh/2)) * (f > ((-BwCh/2) - BwCh))
        
        Pchann[ii] = np.sum(psdy[idx_chann])*BwCh*1e6
        Pabov[ii] = np.sum(psdy[idx_abov])*BwCh*1e6
        Pbelw[ii] = np.sum(psdy[idx_belw])*BwCh*1e6
        
        if OvS > 5:
            #Also the second adjacent channel
            
            idx_abov2 = (f > (BwCh/2 + BwCh)) * (f < ((BwCh/2) + 2*BwCh))
            idx_belw2 = (f < (-BwCh/2) - BwCh) * (f > ((-BwCh/2) - 2*BwCh))
            
            Pabov2[ii] = np.sum(psdy[idx_abov2])*BwCh
            Pbelw2[ii] = np.sum(psdy[idx_belw2])*BwCh
            
    
    ACPR = 10*np.log10(np.mean(np.array([np.mean(Pabov),np.mean(Pbelw)]))/np.mean(Pchann))
    ACPR2 = False
    if OvS > 5:
        ACPR2 = 10*np.log10(np.mean(np.array([np.mean(Pabov2),np.mean(Pbelw2)]))/np.mean(Pchann))
    return ACPR, ACPR2
def EVM_Calculation(const_points, rx_const_points):
    """
    Description
    -----------
    Compute the error vector magnitude (EVM) between the original and resulting point of the constellation

    Parameters
    ----------
    const_points : numpy array complex
        constellation points transmitted.
    rx_const_points : numpy array complex
        constellation points received.

    Returns
    -------
    EVM : float
        EVM performance.

    """
    EVM_Reference = np.amax(np.sqrt(const_points.real**2+const_points.imag**2))
    err = const_points - rx_const_points;
    EVM = np.sqrt(1/err.size*np.sum(err.real**2 + err.imag**2))/EVM_Reference * 100
    return EVM

def NMSE_Calculation(x,y):
    """
    Description
    ----------
    Compute the normalized mean square error for to given values x and y

    Parameters
    ----------
    x : numpy array of complex
        signal 1.
    y : numpy array of complex
        sigtnal 2.

    Returns
    -------
    float
        NMSE.

    """
    return (np.sum(np.abs((x - y))**2))/(np.sum(np.abs(x)**2))

def DrawConstellation(const_points, rx_const_points, NUMPOINTS):
    """
    Description
    ----------
    Draw the constellation with tx and rx points, for a given number of points NUMPOINTS

    Parameters
    ----------
    const_points : numpy array complex
        constellation points transmitted.
    rx_const_points : numpy array complex
        constellation points received.
    NUMPOINTS : int
        Number of points desired for plotting the constellation.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(5,5))
    index = np.random.randint(0, high=len(rx_const_points), size=NUMPOINTS)
    for ii in range(len(index)):
        plt.plot(rx_const_points.real[ii],rx_const_points.imag[ii],'-o',color='blue')
        plt.plot(const_points.real[ii],const_points.imag[ii],'-o',color='red')
    plt.xlabel('I'); plt.ylabel('Q');
    plt.grid(True);
    plt.title('Constellation received')
    plt.savefig('../results/constellationrx.png')
    
def OverSampling_Frequency(X,OvS,Upsampling):
    """
    Description
    ----------
    Perform the oversampling in frequency domain by adding/removing zeros
    
    Parameters
    ----------
    X : numpy array complex
        input signal.
    OvS : float
        Oversampling factor.
    Upsampling : boolean
        True if upsampling must be performed, false if downsampling.

    Returns
    -------
    Y : numpy array complex
        input signal after oversampling.

    """
    #Zero padding OvS times the length of X -> we change fs
    if (OvS != 1):
        (Nx,Nsymb) = np.shape(X)
        if (Upsampling == True):
            #Upsampling: Add zeros
            Ny = np.round(Nx*OvS)
            Y = np.zeros((int(Ny),Nsymb),dtype=np.complex_)
            if Nx % 2 == 0:
                Y.real[int(Ny/2 - np.floor(Nx/2)):int(Ny/2 + np.floor(Nx/2)),:] = X.real
                Y.imag[int(Ny/2 - np.floor(Nx/2)):int(Ny/2 + np.floor(Nx/2)),:] = X.imag
            else:
                Y.real[int(Ny/2 - np.floor(Nx/2)):int(Ny/2 + np.floor(Nx/2)+1),:] = X.real
                Y.imag[int(Ny/2 - np.floor(Nx/2)):int(Ny/2 + np.floor(Nx/2)+1),:] = X.imag
        else:
            #Downsampling: Remove zeros
            Ny = np.round(Nx/OvS)
            Y = np.zeros((int(Ny),Nsymb),dtype=np.complex_)
            if Nx % 2 == 0:
                Y.real = X.real[int(Nx/2 - np.floor(Ny/2)):int(Nx/2 + np.floor(Ny/2)),:]
                Y.imag = X.imag[int(Nx/2 - np.floor(Ny/2)):int(Nx/2 + np.floor(Ny/2)),:]
            else:
                Y.real = X.real[int(Nx/2 - np.floor(Ny/2)):int(Nx/2 + np.floor(Ny/2)+1),:]
                Y.imag = X.imag[int(Nx/2 - np.floor(Ny/2)):int(Nx/2 + np.floor(Ny/2)+1),:]
    else:
        Y = X
    return Y

def RootRaisedCos(Nfilt,B=0,T=1):
    """
    Description
    ----------
    Define root raised cosine function
    
    Parameters
    ----------
    Nfilt : int
        Filter length.
    B : float, optional
        Roll-off factor of root raised cosine function. The default is 0.
    T : float, optional
        Reciprocal of the symbol-rate. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x = np.linspace(-Nfilt/2, Nfilt/2, Nfilt+1)
    return (1/np.sqrt(T))*(np.sin(pi*x/T*(1-B))+4*B*x/T*np.cos(pi*x/T*(1+B)))/(pi*x/T*(1-(4*B*x/T)**2))

def primeFactors(n): 
    """
    Description
    ----------
    Compute the prime factors of a number n

    Parameters
    ----------
    n : int
        input number.

    Returns
    -------
    result : array of float
        prime factors of number n.

    """

    result = np.array([],dtype=np.int)
    # Print the number of two's that divide n 
    while n % 2 == 0: 
        result = np.append(result,2) 
        n = n / 2
    # n must be odd at this point 
    # so a skip of 2 ( i = i + 2) can be used 
    for i in range(3,int(np.sqrt(n))+1,2): 
          
        # while i divides n , print i ad divide n 
        while n % i== 0: 
            result = np.append(result,i) 
            n = n / i 
              
    # Condition if n is a prime 
    # number greater than 2 
    if n > 2: 
       result = np.append(result,n) 
    
    return result

def resample_quotients(fs1, fs2):
    """
    Description
    ----------
    Compute the P and Q resampling coefficients to be used in FFTinterpolate

    Parameters
    ----------
    fs1 : int
        frequency sampling 1.
    fs2 : int
        frequency sampling 2.

    Returns
    -------
    P : float
        P coefficient.
    Q : float
        Q coefficient.

    """
    v1 = primeFactors(fs1)
    v2 = primeFactors(fs2)
    total_ind = np.array([])
    for k in range(0,v1.size):
        #If we can find element k of v1 in v2
        if np.isin(v1[k],v2):
            #Find first index in v2 where it can be found
            ind = np.nonzero(v1[k] == v2)[0][0]
            #Remove the value at index k from v1
            total_ind = np.append(total_ind,k)            
            #Remove the value at index ind from v2
            v2 = np.delete(v2,ind)
            
    P = np.prod(v1[np.setdiff1d(np.arange(0,v1.size),total_ind)])
    Q = np.prod(v2)
    return P,Q

def FFTinterpolate(x,fs_y,fs_u):
    """
    Description
    ----------
    Interpolation performed by FFt algorithm

    Parameters
    ----------
    x : numpy array of complex
        input signal.
    fs_y : float
        sampling rate of the output signal.
    fs_u : TYPE
        sampling rate of the input signal.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    """
    if (fs_u != fs_y):
        N = x.size
        P, Q = resample_quotients(fs_y, fs_u)
        Nn = float(N*P/Q)
        U = np.fft.fft(x)
        Y = np.zeros((int(Nn)),dtype=np.complex_)
        if (np.round(Nn) == Nn):
            Nn = int(Nn)
            Y[Nn-1] = (1j)*10**(-16)
            if (P > Q):
                #Upsampling
                if (Nn % 2) == 0:
                    if (N % 2) == 0:
                        #Even number of samples in u
                        #Easy to put back
                        N = int(N)
                        Y[0:int(N/2)] = U[0:int(N/2)]
                        Y[Nn-int(N/2):Nn] = U[int(N/2):N]
                    else:
                        Y[0:int(np.floor(N/2))] = U[0:int(np.floor(N/2))]
                        Y[Nn-int(np.ceil(N/2)):Nn] = U[int(np.floor(N/2)):N]
                else:
                    print("Not implemented")
                    sys.exit(0)        
            else:
                #Downsampling
                Y[0:int(np.ceil(Nn/2))] = U[0:int(np.ceil(Nn/2))]
                Y[Nn-int(np.ceil(Nn/2)):Nn] = U[N-int(np.ceil(Nn/2)):N]
        else:
            print("Not an integer number of samples. Use some other method")
            sys.exit(0)
        y = np.fft.ifft(Y)
    else:
        y = x
    return y

def Filter_signal(Fs, BwCh, Bw, x):
    """
    Description
    ----------
    Function that filter an input signal using the Kaiser window method.
    Parameters
    ----------
    Fs : float
        frequency sampling.
    BwCh : float
        Channel bandwidth available.
    Bw : float
        bandwidth of the input signal.
    x : numpy array of complex
        input signal.

    Returns
    -------
    y : numpy array of complex
        signal filtered.

    """
    # The Nyquist rate of the signal.
    nyq_freq = Fs / 2.0
    
    Fend = (BwCh / 2) / nyq_freq
    # The cutoff frequency of the filter.
    Fc = (Bw*1.025 / 2) / nyq_freq
    
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = Fend - Fc
    # The desired attenuation in the stop band, in dB.
    ripple_db = 100
    
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    if (N % 2 == 0):
        N = N+1
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, Fc , window=('kaiser', beta))
    delay = int(0.5 * (N))
    # Use lfilter to filter x with the FIR filter.
    #yz = lfilter(taps, 1.0, xz)
    #y = yz[delay:]
    yaux = np.fft.ifft(np.fft.fft(x)*np.fft.fft(taps,x.size))
    y = np.array([])
    y = np.append(y,yaux[delay:])
    y = np.append(y,yaux[0:delay])
    #------------------------------------------------
    # Plot the magnitude response of the filter.
    #------------------------------------------------
    
    plt.figure(2)
    plt.clf()
    w, h = freqz(taps, worN=8000)
    plt.plot((w/pi)*nyq_freq/1e6, 10*np.log10(np.abs(h)), linewidth=2, label="H(f)")
    f, Pxx = PSD(x, Fs, 8000)
    Pxx  = 10*np.log10(Pxx) + 30
    f2, Pyy = PSD(yaux, Fs, 8000)
    Pyy  = 10*np.log10(Pyy) + 30
    plt.plot(f,Pxx - np.max(Pxx),label="X(f)")
    plt.plot(f2,Pyy- np.max(Pyy),label="Y(f)")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.xlabel('frequency [MHz]')
    plt.ylabel('PSD [dBm/Hz]')
    plt.show()
    plt.grid(True);
    plt.title('Filter effect')
    plt.grid(True)
    plt.savefig('../results/filteringx.png')
    return y

def split(Bn, M1, M2):
    k1=int(np.log2(M1))
    k2=int(np.log2(M2))
    k=k1+k2 
    Nb=len(Bn)
    W=np.reshape(Bn,[int(Nb/k),k]) 
    BnI=np.reshape(W[:,:k1],[k1*int(Nb/k)]) 
    BnQ=np.reshape(W[:,k1:],[k2*int(Nb/k)])            
    return BnI,BnQ

def de2gray(d,n):    
    
    gray_list_str = list(GrayCode(n).generate_gray())
    gray_list = list(map(lambda ind: np.array(list(gray_list_str[ind]), dtype=np.int), range(len(gray_list_str))))
    g = list(map(lambda ind: gray_list[int(d[ind])], range(0,len(d))))

    return np.array(g)

def detectsymb(rn, alfabeto):
                
    N = len(rn)

    An = np.zeros(N)
    
    for i in range(N):
        ind = np.where(abs(rn[i]-alfabeto) == np.amin(abs(rn[i]-alfabeto)))
        An[i] = alfabeto[ind]
    
    return An

def gray2de(b):
    """
    Description
    ----------
    Gray to decimal converter
    
    Parameters
    ----------
    b : numpy array of int
        input bits.

    Returns
    -------
    d : numpy array of int
        output bits.

    """
    from numpy import zeros_like, logical_xor, fliplr, shape, reshape, matrix, arange, array
    b=matrix(b)
    c=zeros_like(b)
    c[:,0] = b[:,0];
        
    for i in range(1,shape(b)[1]):
        c[:,i] = logical_xor(c[:,i-1], b[:,i]) 
        
    c=fliplr(c);
    
    [n,m] = shape(c)
    if min([m,n]) < 1:
        d = [];
        return
    elif min([n,m]) == 1:
        m = max([n,m])
        n = 1
        c = reshape(c,[n,m])

    
    d = array((c * matrix(2**arange(m)).T).T).squeeze();
        
    return d

def Qfunct(x):
    """
    Calculation of Q function for BER performance
    ----------

    Parameters
    ----------
    x : numpy array of complex
        input signal.

    Returns
    -------
    res : float
        Q(x).

    """
    res=(1/2)*erfc(x/np.sqrt(2))
    return res

def symb2bit(An,Alphabet):
    """
    Description
    ----------
    Symbol to bit converter

    Parameters
    ----------
    An : numpy array of float
        array of symbols.
    Alphabet : numpy array of float
         alphabet of the QAM modulation.

    Returns
    -------
    Bn : numpy array of int
        sequence of bits corresponding to the input symbols.

    """
    k = np.log2(len(Alphabet))
    
    if k>1:
        dist = abs(Alphabet[0]-Alphabet[1])
        idx = np.round((An-Alphabet[0])/dist)
        Bn = np.reshape(de2gray(idx,k),[int(k*len(An))])
    else:
        Bn = ((An/max(Alphabet))+1)/2
    
    return Bn


    
def qam(Bn, Eb, M1, M2):
    """
    Description
    -----------
    QAM modulator

    Parameters
    ----------
    Bn : numpy array of int
        sequence of bits to transmit.
    Eb : float
        Energy per bit.
    M1 : int
        Modulation order 1.
    M2 : int
        Modulation order 2.

    Returns
    -------
    BnI : numpy array of int
        bits sequence of in-phase component.
    BnQ : numpy array of int
        bit sequence of quadrature component.
    AnI : numpy array of float
        transmitted levels in in-phase component.
    AnQ : numpy array of float
        transmitted levels in quadrature component.
    AI : numpy array of float
        used levels used in in-phase component.
    AQ : numpy array of float
        used levels in quadrature component.

    """

    k1=int(np.ceil(np.log2(M1)))
    M1=2**(k1)              
    k2=int(np.ceil(np.log2(M2)))
    M2=2**(k2)
    
    k=k1+k2
    Nb=len(Bn)
    Bn=np.r_[Bn, np.zeros(int(k*np.ceil(Nb/k)-Nb), dtype=int)]
    
    A= np.sqrt(3*Eb*np.log2(M1*M2)/(M1**2+M2**2-2))
    
    AI=A*(2*np.arange(M1)-M1+1)
    AQ=A*(2*np.arange(M2)-M2+1)
    
    BnI,BnQ=split(Bn,M1,M2)

    NbI=len(BnI)
    NbQ=len(BnQ)

    if M1>2:
        AnI=AI[gray2de(np.reshape(BnI,[int(NbI/k1),k1]))]
    else:
        AnI=AI[BnI]

    
    if M2>2:
        AnQ=AQ[gray2de(np.reshape(BnQ,[int(NbQ/k2),k2]))]  
    else:
        AnQ=AQ[BnQ]
    
    return BnI,BnQ,AnI,AnQ,AI,AQ