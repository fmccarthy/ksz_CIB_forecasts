#cmb and large scale structure power spectra and their experimental noise (to be added)
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import kszpsz_config as conf
import math
import scipy
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import camb
import remote_spectra
import time
#t1=time.time()
import halomodel as halomodel
import galaxies

#import #CIB_Websky

import matplotlib.pyplot as plt

import os
compute_electron_cross = True

savepath="CIB_Cls_not4"
class CmbLssData(object):


    def __init__(self):
        #------ red shift binning (as defined in the config)
        self.zbins_nr = conf.N_bins
        self.zbins_z = remote_spectra.zbins_z
        self.zbins_zcentral = remote_spectra.zbins_zcentral
        self.zbins_chi = remote_spectra.zbins_chi
        self.zbins_chicentral = remote_spectra.zbins_chicentral
        
        #----- useful constants
        #from https://github.com/msyriac/orphics/blob/master/orphics/cosmology.py
        
        self._cSpeedKmPerSec = 299792.458
        self.G_SI = 6.674e-11
        self.mProton_SI = 1.673e-27
        self.H100_SI = 3.241e-18
        self.thompson_SI = 6.6524e-29
        self.meterToMegaparsec = 3.241e-23
        
        self.solar_luminosity_per_Megaparsec2_to_Jansky=self.meterToMegaparsec**2*1e26*(3.827e+26)

        
        #----- CMB
        #convention: l goes from 0 to lmax_global. units are muK (not dim less).
        
        self.lmax_global = conf.estim_smallscale_lmax
        self.ls = np.arange(0,self.lmax_global)
        
        #small scale ksz 
        fname = "data/ClTTksz_halo_latetime_AGN.txt"
        dt = ({ 'names' : ('ls', 'C_l'),'formats' : [np.float, np.float] })
        data = np.loadtxt(fname,dtype=dt)
        l_ksz = data['ls']
        Clksz = data['C_l']
        lin_interp = interp1d(l_ksz,Clksz,bounds_error=False,fill_value=0)
        self.Cl_ksz_TT = lin_interp(self.ls)    
        
       
     
        self.kmax = 100.
        
        self.frequencies=conf.CIB_frequencies
        
        #power spectra calculation
        
        #we can switch halomodel off, and get power spectra from camb with a simple bias prescription
        if not conf.use_halomodel:
            print ("We don't use the halo model. Now calculating CAMB power spectra.")
            #get camb cl power spectrum
            self.cambpars = camb.CAMBparams()
            self.cambpars.set_cosmology(H0=conf.H0, ombh2=conf.ombh2, omch2=conf.omch2)
            self.cambpars.InitPower.set_params(ns=conf.ns, r=0)
            #self.cambpars.set_for_lmax(2500, lens_potential_accuracy=0);
            self.cambpars.set_matter_power(redshifts=self.zbins_z.tolist(), kmax=self.kmax, k_per_logint=20)
            self.cambresults = camb.get_results(self.cambpars) 
            self.PK_nonlin = camb.get_matter_power_interpolator(self.cambpars, nonlinear=True,hubble_units=False, k_hunit=False, kmax=self.kmax, zmax=self.zbins_z[-1])
            print ("Power spectra done.")

        #use halo model
        else:
            #make an halo model object
            print ("We use the halo model. Now calculating Pgg, Pge, Pee power spectra from halo model.")
            self.mthreshHODstellar = galaxies.getmthreshHODstellar(LSSexperiment=conf.LSSexperiment,zbins_central=remote_spectra.zbins_zcentral) #get the right HOD threshold for the experiment, as a function of red shift
            print("starting halomodel")
            self.hmod = halomodel.HaloModel(ukmversion="new",mdef = "200_mean")
            self.logmmin = 8 #we cut by stellar mass, so this essentially integrates over all masses
            self.logmmax = 16
            ks = np.logspace(np.log10(1e-5),np.log10(100.),num=1000) #we evaluate the halo model once on a z and k grid and interpolate below from that.
           # zs = np.linspace(0.1,6,50) #TEST sampling sufficient?
            zs = np.linspace(conf.z_min,conf.z_max,50) #TEST sampling sufficient?
       #     Pgg_sampled = self.hmod.P_gg_1h(ks,zs,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)+self.hmod.P_gg_2h(ks,zs,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)
       #     Pge_sampled = self.hmod.P_ge_1h(ks,zs,gasprofile=conf.gasprofile,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)+self.hmod.P_ge_2h(ks,zs,gasprofile=conf.gasprofile,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)
          #  Pee_sampled = self.hmod.P_ee_1h(ks,zs,gasprofile=conf.gasprofile,logmlow=self.logmmin,logmhigh=self.logmmax)+self.hmod.P_ee_2h(ks,zs,gasprofile=conf.gasprofile,logmlow=self.logmmin,logmhigh=self.logmmax)
         #   Pgm_sampled = self.hmod.P_gm_1h(ks,zs,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)+self.hmod.P_gm_2h(ks,zs,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)
         #   Pmm_sampled = self.hmod.P_mm_1h(ks,zs)+self.hmod.P_mm_2h(ks,zs)

            self.interp_PCIBCIB_1h=[]
            
            self.interp_PCIBCIB_2h=[]
            self.interp_PCIBe=[]
            
            
            
           # self.interp_Pee = interp2d(ks,zs,Pee_sampled,bounds_error=False,fill_value=0)



           # halomodel._setup_k_z_m_grid_functions(zs,mthreshHODstellar)
            
            print("set up")
            
            for frequency_id in range(0,len(self.frequencies)):
                print("doing frequency",frequency_id)
                frequency=self.frequencies[frequency_id]
                t1=time.time()

                #      PCIBCIB_sampled = self.hmod.P_CIBCIB_1h(ks,zs,self.frequency,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)+self.hmod.P_CIBCIB_2h(ks,zs,self.frequency,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)
                PCIBCIB_sampled_1h = self.hmod.P_CIBCIB_1h(ks,zs,frequency,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)
                
                
                print("got one halo in",time.time()-t1,"seconds")
                t1=time.time()
                PCIBCIB_sampled_2h = self.hmod.P_CIBCIB_2h(ks,zs,frequency,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)
                
                np.save(str(frequency)+'_pcibcib2_bothfrs.npy',PCIBCIB_sampled_2h)
                np.save(str(frequency)+'_pcibcib1_bothfrs.npy',PCIBCIB_sampled_1h)

                print("got two halo in",time.time()-t1,"seconds")
                t1=time.time()
                print("getting PCIBe")
                if compute_electron_cross:
                    PCIBe_sampled = self.hmod.P_CIBe_1h(ks,zs,frequency,gasprofile=conf.gasprofile,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)+self.hmod.P_CIBe_2h(ks,zs,frequency,gasprofile=conf.gasprofile,logmlow=self.logmmin,logmhigh=self.logmmax,mthreshHOD=self.mthreshHODstellar)
                print("got cross in",time.time()-t1,"seconds")

            
        
                self.interp_PCIBCIB_1h.append(interp2d(ks,zs,PCIBCIB_sampled_1h,bounds_error=False,fill_value=0))
                self.interp_PCIBCIB_2h.append(  interp2d(ks,zs,PCIBCIB_sampled_2h,bounds_error=False,fill_value=0))
                if compute_electron_cross:
                    self.interp_PCIBe.append( interp2d(ks,zs,PCIBe_sampled,bounds_error=False,fill_value=0))
        
       
            print ("Power spectra done.")

        #now we calculate the binned power spectra that we will need later
     
        
        self.Cl_CIBCIB = np.zeros((3,conf.estim_smallscale_lmax)  )
        self.Cl_CIBCIB_1h = np.zeros((3,conf.estim_smallscale_lmax)  )

        self.Cl_CIBCIB_2h = np.zeros((3,conf.estim_smallscale_lmax) ) 
        
        self.Cl_CIBCIB_binned = np.zeros((3,conf.estim_smallscale_lmax)  )
        self.Cl_CIBCIB_1h_binned = np.zeros((3,conf.zbins_nr,conf.estim_smallscale_lmax)  )

        self.Cl_CIBCIB_2h_binned = np.zeros((3,conf.zbins_nr,conf.estim_smallscale_lmax) ) 
        
        self.Cl_CIBtau = np.zeros((3,conf.zbins_nr,conf.estim_smallscale_lmax) )
        
        self.Cl_tautau = np.zeros( (conf.zbins_nr,conf.estim_smallscale_lmax) )

        
        self.calc_Cl_CIBCIB()
        print("calculated Cl_CIBCIB")
        if compute_electron_cross:
            self.calc_Cl_CIBtau_tautau()
        print("calculated Cl_CIBtau")
        self.Cl_CIBCIB_tot=np.zeros((3,conf.estim_smallscale_lmax)  )
        self.Cl_CIBCIB_tot_binned=np.zeros((3,conf.zbins_nr,conf.estim_smallscale_lmax)  )


        if conf.CIB_model=="minimally_empiricial":
            self.shotnoises=[ self.hmod.CIB_shot_noise(nu)*self.solar_luminosity_per_Megaparsec2_to_Jansky**2 for nu in self.frequencies]
        
        else:
            self.shotnoises=[ self.hmod.CIB_shot_noise(nu)for nu in self.frequencies]

       
            
        if conf.cib_experimental_noise:
            #temperature
            CIB_beams = conf.CIB_beamArcmin_T *np.pi/180./60.
            dT = conf.CIB_noiseTuKArcmin_T *np.pi/180./60.
            with np.errstate(over='ignore'):
                self.Nl_CIB_T = (dT[:,np.newaxis]**2.)*np.exp(self.ls*(self.ls+1.)*(CIB_beams[:,np.newaxis]**2.)/8./np.log(2.))
           
        else:
            self.Nl_CIB_T = [self.ls*0,self.ls*0,self.ls*0]
            
        for i in range(0,len(self.frequencies)):
            
        
            self.Cl_CIBCIB_tot[i,:]=self.Cl_CIBCIB_1h[i,:]+self.Cl_CIBCIB_2h[i,:]+self.shotnoises[i]*0+self.Nl_CIB_T[i][:]
        
            self.Cl_CIBCIB_tot_binned[i,:]=self.Cl_CIBCIB_1h_binned[i,:]+self.Cl_CIBCIB_2h_binned[i,:]#+self.shotnoises[i]+self.Nl_CIB_T[i][:]
            from datetime import date

            today = date.today()            
            direct = (savepath + str(self.frequencies[i]) + "_" + str(conf.N_bins) + "_"  + str(conf.CIB_model)+str(today))
            if not os.path.exists(direct):
                print("making directory")
                os.mkdir(direct)     
                    
            np.save(direct + "/" + "CLs_CIBCIB.npy",self.Cl_CIBCIB_tot[i])
              
        
            print("saved in", direct)
    
    def get_PCIBCIB(self,ks,zs,frequency_id):
       
        PCIBCIB_1h = self.interp_PCIBCIB_1h[frequency_id](ks,zs)[:,::-1].diagonal()
        PCIBCIB_2h = self.interp_PCIBCIB_2h[frequency_id](ks,zs)[:,::-1].diagonal() #need to reverse because of decreasing ks
        #PCIBCIB_1h = self.interp_PCIBCIB_1h[frequency_id](ks,zs).diagonal()
        #PCIBCIB_2h = self.interp_PCIBCIB_2h[frequency_id](ks,zs).diagonal() 
        
        return PCIBCIB_1h,PCIBCIB_2h,#Pee
    
     
    def get_PCIBe_Pee(self,ks,zs,frequency_id):
           
       # PCIBe=self.interp_PCIBe[frequency_id](ks,zs).diagonal()
        PCIBe=self.interp_PCIBe[frequency_id](ks,zs)[:,::-1].diagonal()
      #  Pee = self.interp_Pee(ks,zs)[:,::-1].diagonal()
       # Pee = self.interp_Pee(ks,zs)[:,::-1].diagonal()

        
        return PCIBe
    
   
        
        
    def calc_Cl_CIBCIB(self):
        print ("Calculation Cl_CIBCIB, Cl_CIBtau, Cl_tautau...")
        
        ell_sparse = np.arange(conf.estim_smallscale_lmin,conf.estim_smallscale_lmax,100) #we interpolate these spectra
        
        
        integral_CIBCIB_1h = np.zeros( ell_sparse.shape[0] )

        integral_CIBCIB_2h = np.zeros( ell_sparse.shape[0] )
        
        
        zs = np.linspace(conf.z_min,conf.z_max,100)#change this.

        chi_sampling=remote_spectra.chifromz(zs)
        
      
      
        for frequency_index in range(0,len(self.frequencies)):
            zs = np.linspace(conf.z_min,conf.z_max,100)#change this.

            chi_sampling=remote_spectra.chifromz(zs)
            frequency=self.frequencies[frequency_index]

            for ell_id,ell in enumerate(ell_sparse):
                    #version 1. correct.
                    ks = (ell+(1./2.))/chi_sampling
                   # integrand_chi_CIBtau = np.zeros( chi_sampling.shape[0] )
                    #limber approx
                    PCIBCIB_1h,PCIBCIB_2h = self.get_PCIBCIB(ks,zs,frequency_index)
                   
                
                    if frequency==353:
                    
                        changetomicrokelvinfactor=1/(287.45)
                    
                    elif frequency==545:
                    
                        changetomicrokelvinfactor=1/( 58.04)
                    
                    elif frequency==857:
                        changetomicrokelvinfactor= 1/( 2.27)
                
                    else:
                        print ("frequency not defined at",frequency)
                
                
                    CIBfactor=changetomicrokelvinfactor#self.solar_luminosity_per_Megaparsec2_to_Jansky*changetomicrokelvinfactor

                
                    integrand_CIBCIB_1h =   1/(chi_sampling)**2 * CIBfactor** 2* PCIBCIB_1h
                    integrand_CIBCIB_2h =   1/(chi_sampling)**2 * CIBfactor** 2* PCIBCIB_2h
#                    integrand_CIBtau =   1/(chi_sampling)**2 * electronfactor * CIBfactor * PCIBe
                   
                    integral_CIBCIB_1h[ell_id]= np.trapz(integrand_CIBCIB_1h,chi_sampling)
                    integral_CIBCIB_2h[ell_id]= np.trapz(integrand_CIBCIB_2h,chi_sampling)
                    
                
                
        
              
                #now log interpolate to get all ell
            with np.errstate(divide='ignore'):
                   # plt.loglog(ell_sparse,integral_CIBCIB_2h)
                    #plt.loglog(ell_sparse,integral_CIBCIB_1h)
                    #p#lt.show()
                    lin_interp = scipy.interpolate.interp1d(np.log10(ell_sparse),np.log10(integral_CIBCIB_1h),bounds_error=False,fill_value="extrapolate")          
                    self.Cl_CIBCIB_1h[frequency_index,:] = np.power(10.0, lin_interp(np.log10(self.ls)))
                    self.Cl_CIBCIB_1h[frequency_index,:conf.estim_smallscale_lmin] = 0.
                    lin_interp = scipy.interpolate.interp1d(np.log10(ell_sparse),np.log10(integral_CIBCIB_2h),bounds_error=False,fill_value="extrapolate")          
                    self.Cl_CIBCIB_2h[frequency_index,:] = np.power(10.0, lin_interp(np.log10(self.ls)))
                    self.Cl_CIBCIB_2h[frequency_index,:conf.estim_smallscale_lmin] = 0.
            #'''
            for zbin_id in range(conf.zbins_nr):

                chi_min = remote_spectra.zbins_chi[zbin_id]
                chi_max = remote_spectra.zbins_chi[zbin_id+1]
                #  windownorm = 1./(chi_max-chi_min) #norm of the window function to make it 1 when integrating over chi
                chi_sampling = np.logspace(np.log10(chi_min),np.log10(chi_max),num=100)
                zs = remote_spectra.zfromchi(chi_sampling)
                
                for ell_id,ell in enumerate(ell_sparse):
                    #version 1. correct.
                    ks = (ell+(1./2.))/chi_sampling
                   # integrand_chi_CIBtau = np.zeros( chi_sampling.shape[0] )
                    #limber approx
                    PCIBCIB_1h,PCIBCIB_2h = self.get_PCIBCIB(ks,zs,frequency_index)
                   
                
                    if frequency==353:
                    
                        changetomicrokelvinfactor=1/(287.45)
                    
                    elif frequency==545:
                    
                        changetomicrokelvinfactor=1/( 58.04)
                    
                    elif frequency==857:
                        changetomicrokelvinfactor= 1/( 2.27)
                
                    else:
                        print ("frequency not defined at",frequency)
                
                
                    CIBfactor=changetomicrokelvinfactor#self.solar_luminosity_per_Megaparsec2_to_Jansky*changetomicrokelvinfactor

                
                    integrand_CIBCIB_1h =   1/(chi_sampling)**2 * CIBfactor** 2* PCIBCIB_1h
                    integrand_CIBCIB_2h =   1/(chi_sampling)**2 * CIBfactor** 2* PCIBCIB_2h
#                    integrand_CIBtau =   1/(chi_sampling)**2 * electronfactor * CIBfactor * PCIBe
                   
                    integral_CIBCIB_1h[ell_id]= np.trapz(integrand_CIBCIB_1h,chi_sampling)
                    integral_CIBCIB_2h[ell_id]= np.trapz(integrand_CIBCIB_2h,chi_sampling)
                    
                
                
                with np.errstate(divide='ignore'):
                   # plt.loglog(ell_sparse,integral_CIBCIB_2h)
                    #plt.loglog(ell_sparse,integral_CIBCIB_1h)
                    #p#lt.show()
                    lin_interp = scipy.interpolate.interp1d(np.log10(ell_sparse),np.log10(integral_CIBCIB_1h),bounds_error=False,fill_value="extrapolate")          
                    self.Cl_CIBCIB_1h_binned[frequency_index,zbin_id,:] = np.power(10.0, lin_interp(np.log10(self.ls)))
                    self.Cl_CIBCIB_1h_binned[frequency_index,zbin_id,:conf.estim_smallscale_lmin] = 0.
                    lin_interp = scipy.interpolate.interp1d(np.log10(ell_sparse),np.log10(integral_CIBCIB_2h),bounds_error=False,fill_value="extrapolate")          
                    self.Cl_CIBCIB_2h_binned[frequency_index,zbin_id,:] = np.power(10.0, lin_interp(np.log10(self.ls)))
                    self.Cl_CIBCIB_2h_binned[frequency_index,zbin_id,:conf.estim_smallscale_lmin] = 0.
        
        print ("Calculation Cl_CIBCIB done.")
        
    def calc_Cl_CIBtau_tautau(self):
        print ("Calculation Cl_CIBtau...")
        
        ell_sparse = np.arange(conf.estim_smallscale_lmin,conf.estim_smallscale_lmax,100) #we interpolate these spectra
        
        
        integral_CIBtau = np.zeros( ell_sparse.shape[0])
        
        integral_tautau = np.zeros(ell_sparse.shape[0])

        
        
      
      
        for frequency_index in range(0,len(self.frequencies)):
            
            frequency=self.frequencies[frequency_index]
            
            for zbin_id in range(conf.zbins_nr):

                chi_min = remote_spectra.zbins_chi[zbin_id]
                chi_max = remote_spectra.zbins_chi[zbin_id+1]
                #  windownorm = 1./(chi_max-chi_min) #norm of the window function to make it 1 when integrating over chi
                chi_sampling = np.logspace(np.log10(chi_min),np.log10(chi_max),num=100)
                zs = remote_spectra.zfromchi(chi_sampling)

                for ell_id,ell in enumerate(ell_sparse):
                        #version 1. correct.
                        ks = (ell+(1./2.))/chi_sampling
                        #limber approx
                        PCIBe = self.get_PCIBe_Pee(ks,zs,frequency_index)
                    
                        aas = 1.0/(1.0 + zs)
                
                        if frequency==353:
                    
                            changetomicrokelvinfactor=1.097*1/(287.45)
                    
                        elif frequency==545:
                    
                            changetomicrokelvinfactor=1.068*1/( 58.04)
                    
                        elif frequency==857:
                            changetomicrokelvinfactor= 0.995*1/( 2.27)
                
                        else:
                            print ("frequency not defined at",frequency)
                
                        electronfactor = self.thompson_SI*self.ne0z(zs)*aas**(-2.)/self.meterToMegaparsec *conf.T_CMB #* TcmbMuK depends on if we use dim less or muK CMB
                
                        CIBfactor=changetomicrokelvinfactor#self.solar_luminosity_per_Megaparsec2_to_Jansky*changetomicrokelvinfactor
       
                        integrand_CIBtau =   1/(chi_sampling)**2 * electronfactor * CIBfactor * PCIBe
                      #  integrand_tautau = 1./(chi_sampling**2.) * electronfactor**2. * Pee
                       # integrand_tautau =   1/(chi_sampling)**2 * electronfactor**2  * Pee

                
                        integral_CIBtau[ell_id] = np.trapz(integrand_CIBtau,chi_sampling)
                        
                       # in##tegral_tautau[ell_id] = np.trapz(integrand_tautau,chi_sampling)
#
                
                
              
                #now log interpolate to get all ell
                with np.errstate(divide='ignore'):
                      
                    lin_interp = scipy.interpolate.interp1d(np.log10(ell_sparse),np.log10(integral_CIBtau),bounds_error=False,fill_value="extrapolate") 
                    self.Cl_CIBtau[frequency_index,zbin_id,conf.estim_smallscale_lmin:] = np.power(10.0, lin_interp(np.log10(self.ls[conf.estim_smallscale_lmin:])))
                    self.Cl_CIBtau[frequency_index,zbin_id,:conf.estim_smallscale_lmin] = 0.

                    lin_interp = scipy.interpolate.interp1d(np.log10(ell_sparse),np.log10(integral_tautau),bounds_error=False,fill_value="extrapolate")
                  #  self.Cl_tautau[zbin_id,:] = np.power(10.0, lin_interp(np.log10(self.ls)))
                  #  self.Cl_tautau[zbin_id,:conf.estim_smallscale_lmin] = 0.
                    
                    
                from datetime import date

                today = date.today()            
                direct = (savepath + str(self.frequencies[frequency_index]) + "_" + str(conf.N_bins) + "_"  + str(conf.CIB_model)+str(today))
                if not os.path.exists(direct):
                    print("making directory")
                    os.mkdir(direct)     
                    
                np.save(direct + "/" + "CLs_tauCIB.npy",self.Cl_CIBtau[frequency_index])
                        
                print("saved in", direct)

        print ("Calculation Cl_CIBCIB done.")
                
         
    def ne0z(self,z,shaw=True):
        #from https://github.com/msyriac/orphics/blob/master/orphics/cosmology.py
        '''
        Average electron density today but with
        Helium II reionization at z<3
        Units: 1/meter**3
        '''
        if not(shaw):
            if z>3.: 
                NHe=1.
            else:
                NHe=2.
            ne0_SI = (1.-(4.-NHe)*self.cambpars.YHe/4.)*conf.ombh2 * 3.*(self.H100_SI**2.)/self.mProton_SI/8./np.pi/self.G_SI
        else:
            chi = 0.86
            me = 1.14
            gasfrac = 0.9
            omgh2 = gasfrac* conf.ombh2
            ne0_SI = chi*omgh2 * 3.*(self.H100_SI**2.)/self.mProton_SI/8./np.pi/self.G_SI/me                   
        return ne0_SI
         



#create object that carries our cosmology data
data = CmbLssData()


        
