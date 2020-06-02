#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:23:57 2019

@author: fionamccarthy
"""

#DEPRECATED. from now on we use the ipython notebooks instead of driver.py to call functions.

#python3 setup_CIB.py build_ext --inplace && python3 driver_CIB.py

from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import math
import numpy as np
import ksz_CIB_estimator
#import tensor_forecast
import remote_spectra
#import all_cls
import time
import kszpsz_config as conf


def kszCIB_analysis():
    
    # kszpsz_estimator.read_Cl_vv()
    # kszpsz_estimator.read_Cl_qq_scal()
    # kszpsz_estimator.read_Cl_qq_Etens() #from tensors
    # kszpsz_estimator.read_Cl_qq_Btens() #from tensors

    #plot different Cl spectra
    #kszpsz_estimator.plot_Cl()

    #calc noise PS
    print("about to calculate")
    noisespecs=np.array([conf.beamArcmin_T,conf.noiseTuKArcmin_T,conf.CIB_beamArcmin_T[0],conf.CIB_noiseTuKArcmin_T[0]])#,conf.CIB_beamArcmin_T[1],conf.CIB_noiseTuKArcmin_T[1],conf.CIB_beamArcmin_T[2],conf.CIB_noiseTuKArcmin_T[2]])
    t1=time.time()
    nl,rab=ksz_CIB_estimator.calc_Nl_vv_and_rl_alphabeta_CIB()
    print("CIb reconstruction noise is",nl)
    print("rotation matrix is",rab)
    
    np.savez("noises_CIB/cosmo_params/noise_with_rotation_"+str(conf.N_bins)+"bins_"+str(conf.ksz_estim_signal_lmax)+"_maxestim_all_frequency_"+str(conf.estim_smallscale_lmax)+"_lmax.npz",n_l=nl,r_ab=rab,noise_specs=noisespecs)

    print("number of bins:",conf.N_bins)
    print("lmax",conf.ksz_estim_signal_lmax)
  #  print(" frequency:", conf.frequency_CIB)
    print("max smallscale l",conf.estim_smallscale_lmax)
    print("time taken is",time.time()-t1,"seconds")
    #plot SN
    #kszpsz_estimator.plot_SN_ksz_singlebin()
    #kszpsz_estimator.plot_SN_ksz_multibin()
    #kszpsz_estimator.plot_SN_psz_multibin_fromscalars()
    #kszpsz_estimator.plot_SN_psz_multibin_fromtensors()

    #redshift cov mat and pca
    #kszpsz_signal.calc_redshiftcovmat_signalqq()
    #kszpsz_signal.plot_redshiftcovmat_signalqq()
    #kszpsz_signal.plot_redshiftcovmat_signalvv()
    #kszpsz_signal.calc_PCA(mode="vv")
    #kszpsz_signal.plot_redshiftcorr_signalvv()
    #kszpsz_signal.plot_redshiftcorr_signalqq()
    #kszpsz_signal.calc_redshiftcovmat_signalvv()
    #kszpsz_signal.calc_redshiftcovmat_meansignalvv()
    #kszpsz_signal.plot_signalvv_pointvsmean()

    #r forecast
    #tensor_forecast.calc_r_forecast()
    
    #TEST
     #Cl_qq_Btens = np.loadtxt('/Users/mmunchmeyer/Work/physics/data/kszpszestimator/alex/as_clqq_B_r01_new.txt')
    # print Cl_qq_Btens[0,0]
    # Cl_tens_BB = np.load('/Users/mmunchmeyer/Work/physics/data/kszpszestimator/annesylvie/ClqqB.npy')
    # print Cl_tens_BB[0,0,0]


    

def main():
    kszCIB_analysis()
    
    



if __name__ == "__main__":
    main()

