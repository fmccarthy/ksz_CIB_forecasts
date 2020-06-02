#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:24:21 2019

@author: fionamccarthy
"""

from __future__ import print_function
from __future__ import absolute_import
import sys
sys.path.insert(0, '../Code')
import os
import scipy

from scipy.interpolate import interp2d,interp1d


os.chdir('../Code')

import numpy as np
import matplotlib.pyplot as plt
import time
import camb
from camb import get_matter_power_interpolator, model
from scipy import special

#our modules
import kszpsz_config as conf #here you adjust the parameters you want
import remote_spectra as remote_spectra


planck=6.626e-34
lightspeed=2.99792458e8
kBoltzmann=1.38e-23


def SFR(Mstar,z):
    
    # star formation rate, eq 1 of 1611.04517
    
    t=10.8273*np.arcsinh(1.52753* np.sqrt(1/(1+z)**3)) # age of universe in Gy with omega_lambda=0.7,
                                                       # omega_M=0.3, h=0.7 see after eq 1 in 1611.04517

    return 10**((0.84-0.026 *t)*np.log10(Mstar)-(6.51-0.11*t))

    # in solarmass/year

def TD(z,Mstar): #Dust temperature of starforming galaxy
    return 98 * (1+z)**(-0.065)+6.9*np.log10(SFR(Mstar,z)/Mstar) # equation 9 of 1611.04517


def Planck(nu,T): #Planck law for black body radiation
    return 2*(nu)**3*(1/(np.exp(planck*nu/(kBoltzmann*T))-1))

def SED(nu,T):
    #eq 8 of 1611.04517
    nu=nu*1e9 #multiply by 10^^9 to go from GHz->Hz
    beta=2.1
    normalisation=0.00345432/(kBoltzmann*T/planck)**6.1 # in order that it normalises to 1; 
                                                        # Obviously this normalisation
                                                        # is temperature dependent. This is okay?

    return normalisation*nu**beta*Planck(nu,T) 
def IRX(mstar):
    #see eq 5 of 1611.04517
    alpha=1.5
    IRX0=1.32
    return alpha*np.log10(mstar/10**10.35)+IRX0 #Mstar in solar masses

def LIR(Mstar,z):
    #equation 4 of 1611.04517
    KIR=1.49e-10
    KUV=1.71e-10
    return SFR(Mstar,z)/(KIR+KUV*10**(-IRX(Mstar)))
    # in [SFR]/[KIR]
   

def Lnu(nu,z,Mstar): #spectral luminosity radiance
    Lir=LIR(Mstar,z) #total IR luminosity
    T=TD(z,Mstar) 
    return Lir*SED(nu,T)

def Scentral(nu,z,Mstar):
    #flux from luminosity; eg eq 7 of 1611.04517
    chi=remote_spectra.chifromz(z)
    
    
    return Lnu(nu*(1+z),z,Mstar)/((4*np.pi)*chi**2*(1+z)) # in units of [Lnu]/Mpc**2=solar_luminosity/Mpc**2/Hz

def Luminosity_from_flux(S,z):
    #gives luminosity in [S] * Mpc**2
    return  4 * np.pi * remote_spectra.chifromz(z)**2*(1+z)*S

def subhalo_mass_function(Msub,Mhost):
    #equation 10 from 0909.1325. Need to integrate against ln M. (it gives dn/dlnM_sub)
    return 0.3*(Msub/Mhost)**-0.7*np.exp(-9.9*(Msub/Mhost)**2.5)

def sat_flux(nu,M,z):
    '''
   
    This assigns stellar masses to subhalos according to subhalo mass according to the same relation it assigns
    stellar masses to halos according to subhalo masses: M_S*(M_S)=M*(M) where M_S* is stellar mass of a subhalo,
    M_S is mass of a subhalo, M* is stellar mass of a halo, M is mass of halo. 
    
    '''   
    # M is the halo mass.
    subhalomasses=M.copy()
    
    
  
    
    
    file=np.load("stellarmasses_new.npz")
    mstellars=file["mstellars"]
    precompzs=file["zs"]
    precompms=file["mhalos"]
    mstellar_interp=interp2d(precompms,precompzs,mstellars)
    
    mstellar_am=mstellar_interp(M,z)
   
    integrand=subhalo_mass_function(subhalomasses[:,np.newaxis],M)[:,np.newaxis,:]*Scentral(nu,z[np.newaxis,:],np.transpose(mstellar_am))[:,:,np.newaxis] 
    return np.trapz(integrand,np.log(subhalomasses),axis=0)


fluxcuts=conf.fluxcuts
frequencies=[100,143,217,353,545,857,3000] # in gHz!!
def Scut(nu):
    return fluxcuts[frequencies.index(nu)]

 
 
def shot_noise_chi_integral(nfn,nu,zs,Lcuts,mhalos,stellarmasses):

    Mstars=stellarmasses
 
    integrand=np.zeros(Mstars.shape)
    
   
    integrand=nfn*(Lnu(nu*(1+zs[np.newaxis,:]),zs[np.newaxis,:],Mstars))**2/(4*np.pi)**2 
   
    for i in range(0,integrand.shape[0]):
        for j in range(0,integrand.shape[1]):

            if Lnu(nu*(1+zs[j]),zs[j],Mstars[i,j])>Lcuts[j]:
                integrand[i,j]=0
    return np.trapz(integrand,mhalos,axis=0)
    
    
        

def shot_noise(nfn,nu,zs,mhalos,stellarmasses):
    chis=remote_spectra.chifromz(zs) 

    Lcuts=Luminosity_from_flux(Scut(nu),zs)  #Scut is in mJy=1e-29W/m**2/Hz. This Lcut will be in mJy Mpc**2
    Lcuts= Lcuts *(3.086e22)**2  #now in mJy m**2 = 1e-29 W/Hz
    Lcuts=Lcuts*1e-29  #in W/Hz
    Lcuts=Lcuts/3.8e26  #insolarluminosities per hz
    integrand=1/remote_spectra.chifromz(zs)**2*1/(1+zs)**2*shot_noise_chi_integral(nfn,nu,zs,Lcuts,mhalos,stellarmasses) 
                                    #in 1/MPc**2*[shot_noise_chi_integral]=1/MPc**2*[stellar_mass_function]*L**2 = Mpc**-5*[L]**2=
                                    #=Mpc**-5* [solar luminosity/Hz]**2
    
    shot_noise=np.trapz(integrand,chis) #in MPc*[integrand]=1/Mpc*[shot_noise_chi_integral]=Mpc**-4*[L]**2=
        
    if nu==353:
                    
        changetomicrokelvinfactor=1*1/(287.45)
                    
    elif nu==545:
                    
        changetomicrokelvinfactor=1*1/( 58.04)
                    
    elif nu==857:
        changetomicrokelvinfactor= 1*1/( 2.27)
                
                
    return shot_noise*changetomicrokelvinfactor**2

    
        
    
        
