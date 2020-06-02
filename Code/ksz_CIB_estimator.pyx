#this file calculates the forecasts for the kszpsz estimator paper. ie. it calculates eg Nl_vv, Nl_qq.
#The signal Cl_vv is only read in from a file. Signal calculation should happen in kszpsz_signal.py 

from __future__ import print_function
from __future__ import absolute_import

cimport cython
from cython.parallel import parallel, prange
from libcpp.vector cimport vector
from libc.stdio cimport *
from libc.math cimport sqrt
from libc.math cimport pow as cpow
from libc.math cimport log
from libc.math cimport exp as cexp
from libc.math cimport sin as csin
from libc.math cimport floor as cfloor
from libc.math cimport cos as ccos
from libc.math cimport fabs as cabs
from libc.math cimport log as clog
from libc.math cimport sqrt as csqrt
import math
import numpy as np
from scipy.special import jn
from scipy.special import spherical_jn
import scipy.sparse
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from sympy.physics.wigner import wigner_3j
import pywigxjpf as wig
cdef double pi = math.pi

import kszpsz_config as conf
import all_cls
import kszpsz_signal
import remote_spectra


############################# calculate Cl and Nl  #########################


Cl_vv = []
#read from cov mat calculated below
def read_Cl_vv():
    global Cl_vv
    Cl_vv = np.zeros( (conf.zbins_nr,conf.ksz_estim_signal_lmax) ) #zbins_id is the bin index, starts at 0. ell also starts at 0
    #read in my file
    signal_cov_mat_vv = np.load(conf.data_path+'forecast/cov_vv_mean_'+str(conf.zbins_nr)+'bins.npy') #cov_vv_ cov_vv_mean_
    for zbins_id in range(conf.zbins_nr):
        Cl_vv[zbins_id] = signal_cov_mat_vv[zbins_id,zbins_id,:] 
        
 
'''
def calc_Nl_vv_CIB():#Fiona
    print ("Calculating Nl_vv...")
    #setup wigners
    maxfact = conf.estim_smallscale_lmax + conf.estim_smallscale_lmax + conf.ksz_estim_signal_lmax + 1
    calc_logfactlist(maxfact)
    #calc the noise
    Nl_vv = np.zeros( (3,conf.zbins_nr,conf.zbins_nr,conf.ksz_estim_signal_lmax) ) #zbins_id is the bin index, starts at 0. ell also starts at 0
    
    Rl_alphabeta=np.zeros( (3,conf.zbins_nr,conf.zbins_nr,conf.ksz_estim_signal_lmax) ) 
    
    calc_Nl_vv_worker_CIB(Nl_vv,Rl_alphabeta,all_cls.data.Cl_TT_tot,all_cls.data.Cl_CIBCIB_tot,all_cls.data.Cl_CIBtau,conf.ksz_estim_signal_lmin,conf.ksz_estim_signal_lmax,conf.estim_smallscale_lmin,conf.estim_smallscale_lmax)
    # np.save(conf.data_path+'forecast/NoiseHalo/Nl_vv_'+str(conf.estim_smallscale_lmax-1)+'_N='+str(conf.N_bins)+'npy', Nl_vv)
#    np.save(conf.data_path+'forecast/Cl_TT_tot.npy', cmblss.data.Cl_TT_tot)
#    np.save(conf.data_path+'forecast/Cl_gg_tot.npy', cmblss.data.Cl_gg_tot)
#    np.save(conf.data_path+'forecast/Cl_gtau.npy', cmblss.data.Cl_gtau)
    #print ("All done.")
    return Nl_vv
'''  
'''
cdef calc_Nl_vv_worker_CIB(double [:,:,:,:] Nl_vv,double[:,:,:,:] Rl_alphabeta,double [:] Cl_TT, double [:,:] Cl_CIBCIB, double [:,:,:] Cl_CIBtau, int ksz_estim_signal_lmin,int ksz_estim_signal_lmax,int estim_smallscale_lmin,int estim_smallscale_lmax):
    cdef double terms_numerator[:],terms_denom_alpha[:],terms_denom_beta[:], gamma_g_tau_alpha[:],gamma_g_tau_beta[:], fourpi, wigner,terms_rl[:]
    fourpi = 4.0*np.pi
    
#    for frequency_index in range(0,3):
 #       print("frequency is", frequency_index)
        
        for zbins_id_alpha in range(conf.zbins_nr):
            print("zbin is",zbins_id_alpha)

            for zbins_id_beta in range(1):
                print("zbin is",zbins_id_beta)
                zbins_id_beta=zbins_id_alpha
                for ell_id in range(ksz_estim_signal_lmin,ksz_estim_signal_lmax):
                        print ("ell is",ell_id,"for alpha index=",zbins_id_alpha,"beta=",zbins_id_beta,"frequency =",frequency_index)

                        terms_numerator = [0,0,0]
                        terms_denom_alpha=[0,0,0]
                        terms_denom_beta=[0,0,0] 
                        terms_rl=[0,0,0]
                        print(estim_smallscale_lmin,estim_smallscale_lmax)
                        for ell_1 in range(estim_smallscale_lmin,estim_smallscale_lmax):
                            print(ell_1)
                            print("in")
                            for ell_2 in range(estim_smallscale_lmin,estim_smallscale_lmax):
                                print(ell_2)
                                if not ((cabs(ell_1-ell_2) <= ell_id) and (ell_id <= ell_1+ell_2)): #triangle rule
                                    continue
                            #if not ((ell_1+ell_2+ell_id)%2==0):
                            #    continue #parity rule
                            
                            
                            
                                #first the nl_vv:
                                #if zbins_id_beta>=zbins_id_alpha:

                                wigner = wigner_3j_m0(ell_id,ell_1,ell_2)
                                #possibly precompute this? although this is 40*2000*2000....
                                #possibly ask it to only compute half of ell_2 (wigner 3j is 0 for ell_1+ell_2+ell_id = 2n+1)
                                for frequency_index in range(0,3):
                                
                                    gamma_g_tau_alpha[frequency_index] = np.sqrt((2.*ell_id+1.))*np.sqrt((2.*ell_1+1.))*np.sqrt((2.*ell_2+1.))*np.sqrt(1.0/fourpi) * wigner * Cl_CIBtau[frequency_index,zbins_id_alpha,ell_2]
                                    gamma_g_tau_beta[frequency_index] = np.sqrt((2.*ell_id+1.))*np.sqrt((2.*ell_1+1.))*np.sqrt((2.*ell_2+1.))*np.sqrt(1.0/fourpi) * wigner * Cl_CIBtau[frequency_index,zbins_id_beta,ell_2]
 
                                    terms_numerator[frequency_index] += gamma_g_tau_alpha[frequency_index]*gamma_g_tau_beta[frequency_index]/(Cl_TT[ell_1]*Cl_CIBCIB[frequency_index,ell_2])
                                
                                    terms_denom_alpha[frequency_index]+=gamma_g_tau_alpha[frequency_index]**2./(Cl_TT[ell_1]*Cl_CIBCIB[frequency_index,ell_2])
                        
                                    terms_denom_beta[frequency_index]+=gamma_g_tau_beta**2./(Cl_TT[ell_1]*Cl_CIBCIB[frequency_index,ell_2])

                                    if math.isnan(terms_numerator[frequency_index]):
                                        print(1.0/fourpi)
                                #next the Rl_alphabeta:
                        
                        
                        for frequency_index in range(0,3):            
                            print("tn",1./(terms_denom_alpha*terms_denom_beta)*terms_numerator)
                            print("tot" , ((2.*ell_id+1.))*1./(terms_denom_alpha*terms_denom_beta)*terms_numerator)
                            Nl_vv[frequency_index,zbins_id_alpha,zbins_id_beta,ell_id]=   ((2.*ell_id+1.))*1./(terms_denom_alpha[frequency_index]*terms_denom_beta[frequency_index])*terms_numerator[frequency_index]
                            Rl_alphabeta[frequency_index,zbins_id_alpha,zbins_id_beta,ell_id]=1./terms_denom_alpha[frequency_index]*terms_numerator[frequency_index]
            #print (Nl_vv[zbins_id,ell_id])
'''

def calc_Nl_vv_and_rl_alphabeta_CIB():#Fiona
    print ("Calculating Nl_vv...")
    #setup wigners
    maxfact = conf.estim_smallscale_lmax + conf.estim_smallscale_lmax + conf.ksz_estim_signal_lmax + 1
    calc_logfactlist(maxfact)
    #calc the noise
    Nl_vv = np.zeros( (3,conf.zbins_nr,conf.zbins_nr,conf.ksz_estim_signal_lmax) ) #zbins_id is the bin index, starts at 0. ell also starts at 0
    
    Rl_alphabeta=np.zeros( (3,conf.zbins_nr,conf.zbins_nr,conf.ksz_estim_signal_lmax) ) 
    
    calc_Nl_vv_and_Rlalphabeta_worker_CIB(Nl_vv,Rl_alphabeta,all_cls.data.Cl_TT_tot,all_cls.data.Cl_CIBCIB_tot,all_cls.data.Cl_CIBtau,conf.ksz_estim_signal_lmin,conf.ksz_estim_signal_lmax,conf.estim_smallscale_lmin,conf.estim_smallscale_lmax)
    # np.save(conf.data_path+'forecast/NoiseHalo/Nl_vv_'+str(conf.estim_smallscale_lmax-1)+'_N='+str(conf.N_bins)+'npy', Nl_vv)
#    np.save(conf.data_path+'forecast/Cl_TT_tot.npy', cmblss.data.Cl_TT_tot)
#    np.save(conf.data_path+'forecast/Cl_gg_tot.npy', cmblss.data.Cl_gg_tot)
#    np.save(conf.data_path+'forecast/Cl_gtau.npy', cmblss.data.Cl_gtau)
    #print ("All done.")
    return Nl_vv,Rl_alphabeta
    
cdef calc_Nl_vv_and_Rlalphabeta_worker_CIB(double [:,:,:,:] Nl_vv,double[:,:,:,:] Rl_alphabeta,double [:] Cl_TT, double [:,:] Cl_CIBCIB, double [:,:,:] Cl_CIBtau, int ksz_estim_signal_lmin,int ksz_estim_signal_lmax,int estim_smallscale_lmin,int estim_smallscale_lmax):
    cdef double gamma_g_tau_alpha_freq_independent,gamma_g_tau_beta_freq_independent, fourpi, wigner,terms_rl
    cdef double terms_numerator[3]
    cdef double terms_denom_alpha[3]
    cdef double terms_denom_beta[3]
    cdef double gamma_g_tau_alpha[3]
    cdef double gamma_g_tau_beta[3]
    cdef double Aalpha[3]
    cdef double Abeta[3]
    fourpi = 4.0*np.pi
    
    #for frequency_index in range(0,3):
     #   print("frequency is", frequency_index)
        
    for zbins_id_alpha in range(conf.zbins_nr):
            print("zbin is",zbins_id_alpha)

            for zbins_id_beta in range(conf.zbins_nr):
                print("zbin is",zbins_id_beta)
    
                for ell_id in range(ksz_estim_signal_lmin,ksz_estim_signal_lmax):
                        print ("ell is",ell_id,"for alpha index=",zbins_id_alpha,"beta=",zbins_id_beta)

                        terms_numerator = [0,0,0]
                        terms_denom_alpha=[0,0,0]
                        terms_denom_beta=[0,0,0] 
                        terms_rl=0
                        
                        for ell_1 in range(estim_smallscale_lmin,estim_smallscale_lmax):
                          #  print(ell_1)
                            for ell_2 in range(estim_smallscale_lmin,estim_smallscale_lmax):
                                if not ((cabs(ell_1-ell_2) <= ell_id) and (ell_id <= ell_1+ell_2)): #triangle rule
                                    continue
                            #if not ((ell_1+ell_2+ell_id)%2==0):
                            #    continue #parity rule
                            
                              
                                #first the nl_vv:
                                #if zbins_id_beta>=zbins_id_alpha:

                                wigner = wigner_3j_m0(ell_id,ell_1,ell_2)
                                gamma_g_tau_alpha_freq_independent = np.sqrt((2*ell_id+1))*np.sqrt((2*ell_1+1))*np.sqrt((2*ell_2+1))*np.sqrt(1.0/fourpi) * wigner 
                                gamma_g_tau_beta_freq_independent = np.sqrt((2*ell_id+1))*np.sqrt((2*ell_1+1))*np.sqrt((2*ell_2+1))*np.sqrt(1.0/fourpi) * wigner 
 
                                for frequency_index in range(0,3):
                                #possibly precompute this? although this is 40*2000*2000....
                                #possibly ask it to only compute half of ell_2 (wigner 3j is 0 for ell_1+ell_2+ell_id = 2n+1)
                                
                                    gamma_g_tau_alpha [frequency_index]= gamma_g_tau_alpha_freq_independent * Cl_CIBtau[frequency_index,zbins_id_alpha,ell_2]
                                    gamma_g_tau_beta [frequency_index]= gamma_g_tau_beta_freq_independent * Cl_CIBtau[frequency_index,zbins_id_beta,ell_2]
                                                
                                    terms_numerator[frequency_index] += gamma_g_tau_alpha[frequency_index]*gamma_g_tau_beta[frequency_index]/(Cl_TT[ell_1]*Cl_CIBCIB[frequency_index,ell_2])
                                
                                    terms_denom_alpha[frequency_index] += gamma_g_tau_alpha[frequency_index]**2./(Cl_TT[ell_1]*Cl_CIBCIB[frequency_index,ell_2])
                        
                                    terms_denom_beta[frequency_index]+= gamma_g_tau_beta[frequency_index]**2./(Cl_TT[ell_1]*Cl_CIBCIB[frequency_index,ell_2])
                            #    print("summing",frequency_index,zbins_id_alpha,zbins_id_beta,ell_id,ell_1,ell_2,wigner,gamma_g_tau_alpha,terms_denom_alpha, Cl_CIBtau[frequency_index,zbins_id_alpha,ell_2])
                                    if math.isnan(terms_numerator[frequency_index]):
                                        print(1.0/fourpi)
                                #next the Rl_alphabeta:
                      #  print("final",terms_denom_alpha)
                        for frequency_index in range(0,3):  
                            print(                        (Cl_CIBtau[frequency_index,zbins_id_alpha,:]))
                            print(ell_id,frequency_index,terms_denom_alpha[frequency_index])
                            #print("terms denom alpha",frequency_index,terms_denom_alpha[0])
                            Aalpha[frequency_index]=   (2.*ell_id+1.)/   terms_denom_alpha[frequency_index]
                            Abeta[frequency_index]= (2.*ell_id+1.)/   terms_denom_beta[frequency_index]

                      #  Nl_vv[frequency_index,zbins_id_alpha,zbins_id_beta,ell_id]=   (2.*ell_id+1.)**2./(terms_denom_alpha*terms_denom_beta)*terms_numerator
                            Nl_vv[frequency_index,zbins_id_alpha,zbins_id_beta,ell_id]=   1./(2.*ell_id+1.)*terms_numerator[frequency_index]*Aalpha[frequency_index]*Abeta[frequency_index]
                       # Rl_alphabeta[frequency_index,zbins_id_alpha,zbins_id_beta,ell_id]=1/terms_denom_alpha*terms_numerator
                            Rl_alphabeta[frequency_index,zbins_id_alpha,zbins_id_beta,ell_id]=terms_numerator[frequency_index]/(2.*ell_id+1.)*Aalpha[frequency_index]
            #print (Nl_vv[zbins_id,ell_id])




  
############################# SN forecast plots (now in jupyter)  #########################
            
# def plot_SN_ksz_singlebin():
#     zbins_id = 2
    
#     Nl_vv = np.load(conf.data_path+'forecast/Nl_vv_'+str(conf.estim_smallscale_lmax-1)+'.npy')
#     ells = range(conf.ksz_estim_signal_lmin,conf.ksz_estim_signal_lmax)
#     Nl_vv_bin = Nl_vv[zbins_id,conf.ksz_estim_signal_lmin:conf.ksz_estim_signal_lmax]
#     Cl_vv_bin = Cl_vv[zbins_id,conf.ksz_estim_signal_lmin:conf.ksz_estim_signal_lmax]
#     SN_vv_bin = np.sqrt(1./2. * (Cl_vv_bin/Nl_vv_bin)**2.)

#     fig=plt.figure(figsize=(6,4))
#     ax1 = fig.add_subplot(111)
#     ax1.plot(ells,SN_vv_bin,color='blue',ls='solid',label=r'$S/N$')
#     #ax1.set_xlim(2,2000)
#     ax1.set_yscale('log')
#     #ax1.set_ylabel(r'$B_{lll}$', fontsize=16)
#     #ax1.set_xlabel(r'$\ell$', fontsize=16)
#     #ax1.set_ylabel(r'$b_{\ell \ell \ell}/b_{\ell \ell \ell}^{const}$', fontsize=16)
#     #plt.legend(loc=1,frameon=False) #1:right top. 2: left top. 4: right bottom   
#     fig.tight_layout()
#     plt.show()
#     fig.savefig(conf.data_path+'plots/SN.png')


# def plot_SN_ksz_multibin():
#     Nl_vv = np.load(conf.data_path+'forecast/Nl_vv_'+str(conf.estim_smallscale_lmax-1)+'.npy')
#     ells = range(conf.ksz_estim_signal_lmin,conf.ksz_estim_signal_lmax)
#     fig=plt.figure(figsize=(6,4))
#     ax1 = fig.add_subplot(111)
    
#     for zbins_id in range(conf.zbins_nr):
#         Nl_vv_bin = Nl_vv[zbins_id,conf.ksz_estim_signal_lmin:conf.ksz_estim_signal_lmax]
#         Cl_vv_bin = Cl_vv[zbins_id,conf.ksz_estim_signal_lmin:conf.ksz_estim_signal_lmax]
#         SN_vv_bin = np.sqrt(1./2. * (Cl_vv_bin/Nl_vv_bin)**2.) #kendrick
#         #SN_vv_bin = np.sqrt(Cl_vv_bin/Nl_vv_bin) #ours
#         ax1.plot(ells,SN_vv_bin,ls='solid',label='bin' + str(zbins_id+1)) #,color='blue'
        
#     #ax1.set_xlim(2,2000)
#     #ax1.set_ylim(2,2000)
#     plt.axhline(y=1.,color='k',ls='dashed')
#     ax1.set_yscale('log')
#     ax1.set_ylabel(r'${\rm SN}_{\ell m}$', fontsize=16)
#     ax1.set_xlabel(r'$\ell$', fontsize=16)
#     ax1.set_xlim(1.,10.)
#     plt.xticks(range(1,10)) 
#     #ax1.set_ylabel(r'$b_{\ell \ell \ell}/b_{\ell \ell \ell}^{const}$', fontsize=16)
#     #plt.legend(loc=1,frameon=False) #1:right top. 2: left top. 4: right bottom   
#     fig.tight_layout()
#     plt.show()
#     fig.savefig(conf.data_path+'plots/SN_'+str(conf.ksz_estim_signal_lmin)+'_'+str(conf.ksz_estim_signal_lmax)+'_vv.pdf') 


    
# def plot_SN_psz_multibin_fromscalars():
#     Nl_qq = np.load(conf.data_path+'forecast/Nl_qq_E_'+str(conf.estim_smallscale_lmax-1)+'.npy')
#     ells = range(conf.psz_estim_signal_lmin,conf.psz_estim_signal_lmax)
#     fig=plt.figure(figsize=(6,4))
#     ax1 = fig.add_subplot(111)
    
#     for zbins_id in range(conf.zbins_nr):
#         Nl_qq_bin = Nl_qq[zbins_id,conf.psz_estim_signal_lmin:conf.psz_estim_signal_lmax]
#         Cl_qq_bin = Cl_qq_scal[zbins_id,conf.psz_estim_signal_lmin:conf.psz_estim_signal_lmax]
#         SN_qq_bin = np.sqrt(1./2. * (Cl_qq_bin/Nl_qq_bin)**2.) #kendricks
#         #SN_qq_bin = np.sqrt(Cl_qq_bin/Nl_qq_bin) #ours
#         #SN_qq_bin = Cl_qq_bin/Nl_qq_bin #like normal powers
#         ax1.plot(ells,SN_qq_bin,ls='solid',label='bin' + str(zbins_id+1)) #,color='blue'
        
#     ax1.set_xlim(2,10)
#     #ax1.set_ylim(2,2000)
#     plt.axhline(y=1.,color='k',ls='dashed')
#     ax1.set_yscale('log')
#     ax1.set_ylabel(r'${\rm SN}_{\ell m}$', fontsize=16)
#     ax1.set_xlabel(r'$\ell$', fontsize=16)
#     plt.legend(loc=1,frameon=False) #1:right top. 2: left top. 4: right bottom   
#     fig.tight_layout()
#     plt.show()
#     fig.savefig(conf.data_path+'plots/SN_'+str(conf.psz_estim_signal_lmin)+'_'+str(conf.psz_estim_signal_lmax)+'_qq_E_fromscalars.pdf') 
    

    

# def plot_SN_psz_multibin_fromtensors():
#     Nl_qq_E = np.load(conf.data_path+'forecast/Nl_qq_E_'+str(conf.estim_smallscale_lmax-1)+'.npy')
#     Nl_qq_B = np.load(conf.data_path+'forecast/Nl_qq_B_'+str(conf.estim_smallscale_lmax-1)+'.npy')
#     ells = range(conf.psz_estim_signal_lmin,conf.psz_estim_signal_lmax)

#     fig=plt.figure(figsize=(6,4))
#     ax1 = fig.add_subplot(111)  
#     for zbins_id in range(conf.zbins_nr):
#         Nl_qq_E_bin = Nl_qq_E[zbins_id,conf.psz_estim_signal_lmin:conf.psz_estim_signal_lmax]
#         Cl_qq_Etens_bin = Cl_qq_Etens[zbins_id,conf.psz_estim_signal_lmin:conf.psz_estim_signal_lmax]
#         SN_qq_E_bin = np.sqrt(1./2. * (Cl_qq_Etens_bin/Nl_qq_E_bin)**2.) #like kendrick
#         #SN_qq_E_bin = np.sqrt(Cl_qq_Etens_bin/Nl_qq_E_bin) #like amplitudes
#         #SN_qq_E_bin = Cl_qq_Etens_bin/Nl_qq_E_bin #like normal powers
#         ax1.plot(ells,SN_qq_E_bin,ls='solid',label='bin' + str(zbins_id+1)) #,color='blue'

#     ax1.set_ylim(10.**(-3.),100.)
#     ax1.set_xlim(2,10)
#     ax1.axhline(y=1.,color='k',ls='dashed')
#     ax1.set_yscale('log')
#     ax1.set_ylabel(r'${\rm SN}_{\ell m}$', fontsize=16)
#     ax1.set_xlabel(r'$\ell$', fontsize=16)
#     #ax1.set_title("E-mode quadrupole field from tensors")
#     ax1.legend(loc=1,frameon=False) #1:right top. 2: left top. 4: right bottom   
#     fig.tight_layout()
#     plt.show()
#     fig.savefig(conf.data_path+'plots/SN_sigmaE_EB_fromtensors_lmax'+str(conf.estim_smallscale_lmax-1)+'.pdf')

    
        
#     fig=plt.figure(figsize=(6,4))
#     ax2 = fig.add_subplot(111)  
#     for zbins_id in range(conf.zbins_nr):     
#         Nl_qq_B_bin = Nl_qq_B[zbins_id,conf.psz_estim_signal_lmin:conf.psz_estim_signal_lmax]
#         Cl_qq_Btens_bin = Cl_qq_Btens[zbins_id,conf.psz_estim_signal_lmin:conf.psz_estim_signal_lmax]
#         SN_qq_B_bin = np.sqrt(1./2. * (Cl_qq_Btens_bin/Nl_qq_B_bin)**2.) #like kendrick
#         #SN_qq_B_bin = np.sqrt(Cl_qq_Btens_bin/Nl_qq_B_bin) #like amplitudes
#         #SN_qq_B_bin = Cl_qq_Btens_bin/Nl_qq_B_bin #like normal powers
#         ax2.plot(ells,SN_qq_B_bin,ls='solid',label='bin' + str(zbins_id+1)) #,color='blue'        

#     ax2.set_ylim(10.**(-3.),100.)
#     ax2.set_xlim(2,10)
#     ax2.axhline(y=1.,color='k',ls='dashed')
#     ax2.set_yscale('log')
#     ax2.set_ylabel(r'${\rm SN}_{\ell m}$', fontsize=16)
#     ax2.set_xlabel(r'$\ell$', fontsize=16)
#     #ax2.set_title("B-mode quadrupole field from tensors")
#     ax2.legend(loc=1,frameon=False) #1:right top. 2: left top. 4: right bottom 
#     fig.tight_layout()
#     plt.show()
#     fig.savefig(conf.data_path+'plots/SN_sigmaB_EB_fromtensors_lmax'+str(conf.estim_smallscale_lmax-1)+'.pdf') 



# def plot_Cl():

#     # #Cl TT
#     # ells = np.arange(conf.estim_smallscale_lmax)
#     # fig=plt.figure(figsize=(6,4))
#     # ax1 = fig.add_subplot(111)
#     # ax1.plot(ells,cmblss.data.Cl_TT*ells*(ells+1)/(2.*math.pi),color='blue',ls='solid')
#     # ax1.set_yscale('log')
#     # ax1.set_xscale('log')
#     # fig.tight_layout()
#     # #plt.show()
#     # #fig.savefig(conf.data_path+'plots/ClTT.png')

#     #Cl EE and BB
#     # ells = np.arange(conf.estim_smallscale_lmax)
#     # fig=plt.figure(figsize=(6,4))
#     # ax1 = fig.add_subplot(111)
#     # ax1.plot(ells,cmblss.data.Cl_EE*ells*(ells+1)/(2.*math.pi),color='blue',ls='solid')
#     # ax1.plot(ells,cmblss.data.Cl_BB*ells*(ells+1)/(2.*math.pi),color='red',ls='solid')
#     # ax1.set_yscale('log')
#     # ax1.set_xscale('log')
#     # fig.tight_layout()
#     # plt.show()
#     #fig.savefig(conf.data_path+'plots/ClTT.png')  

#     # #Cl gg
#     # ells = np.arange(conf.estim_smallscale_lmax)
#     # zbins_id = 2
#     # fig=plt.figure(figsize=(6,4))
#     # ax1 = fig.add_subplot(111)
#     # ax1.plot(ells,cmblss.data.Cl_gg[zbins_id]*ells*(ells+1)/(2.*math.pi),color='blue',ls='solid')
#     # ax1.set_yscale('log')
#     # ax1.set_xscale('log')
#     # fig.tight_layout()
#     # #plt.show()
#     # #fig.savefig(conf.data_path+'plots/Clgg.png')

#     # #Cl vv
#     ells = np.arange(conf.ksz_estim_signal_lmax)
#     zbins_id = 2
#     fig=plt.figure(figsize=(6,4))
#     ax1 = fig.add_subplot(111)
#     ax1.plot(ells[2:],Cl_vv[zbins_id][2:],color='blue',ls='solid')
#     ax1.set_yscale('log')
#     ax1.set_xscale('log')
#     fig.tight_layout()
#     plt.show()
#     #fig.savefig(conf.data_path+'plots/Clvv.png')

#     #Clqq
#     # ells = np.arange(conf.psz_estim_signal_lmax)
#     # zbins_id = 2
#     # fig=plt.figure(figsize=(6,4))
#     # ax1 = fig.add_subplot(111)
#     # ax1.plot(ells[2:],Cl_qq_scal[zbins_id][2:],color='blue',ls='solid') #*ells[2:]*(ells[2:]+1)/(2.*math.pi)
#     # ax1.set_yscale('log')
#     # ax1.set_xscale('log')
#     # fig.tight_layout()
#     # plt.show()




############################# Wigner stuff  #########################



cdef vector[double] logfactlist 

def calc_logfactlist(maxfact):
    print ("Setting up factorial table.")
    cdef double one = 1.0
    if (len(logfactlist)==0):
        logfactlist.push_back(0)
        for i in range(maxfact):
            logfactlist.push_back(logfactlist[i]+clog(i+one))
    print ("Done.")


cdef double logfactorial(int n): #nogil
    return logfactlist[n]

cdef double wigner_3j_m0(int j_1,int j_2,int j_3): #nogil  #NOTE THAT j123 must fullfill triangle rule, it is not tested here!
    cdef int J = j_1 + j_2 + j_3
    cdef int g = 0
    cdef double result = 0
    if J%2 != 0:
        return 0
    else:  
        g = J/2
        result = (1./2.)*(logfactorial(2*g-2*j_1) + logfactorial(2*g-2*j_2) + logfactorial(2*g-2*j_3) - logfactorial(2*g+1))
        result += logfactorial(g) - (logfactorial(g-j_1)+logfactorial(g-j_2)+logfactorial(g-j_3))
        result = cpow((-1),g)*cexp(result)
        return result
