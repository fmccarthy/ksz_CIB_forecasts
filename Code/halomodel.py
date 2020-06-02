# Preamble. 
# Make sure you are running CAMB version 0.1.6.1
from __future__ import print_function
from __future__ import absolute_import
import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.integrate as integrate
from scipy.interpolate import interp2d,interp1d
import camb
import configparser as ConfigParser
import kszpsz_config

import time
print("reloading halomodel")
import remote_spectra

import CIB#,CIB_Planck,CIB_Websky

mthreshHODdefault = np.array( [10.5] ) #only used if not passed as z-array


class HaloModel(object):

    def __init__(self,logmmin=8,logmmax=16,npts=1000,kpoints=2000,mdef = None,ukmversion="old"):
        #flags
        print("starting halomodel")
        print("ukm version is",ukmversion)
        self.onehdamping = False #damping envelope
        self.onehdamping_kstar = 0.03
        
        self.mdef = mdef #mass definition is SO mass for a region with mean density 200 times matter density. 

     
        # Setup the cosmology, range of halo mass, and collapse model.

        #--------- Define the cosmology (TODO replace by Bint)

        self.ombh2 = kszpsz_config.ombh2        
        self.omch2 = kszpsz_config.omch2        
        self.omegam = kszpsz_config.Omega_m
        self.As = kszpsz_config.As
        self.ns = kszpsz_config.ns
        self.H0 = kszpsz_config.H0
        self.h = kszpsz_config.h 
        self.cspeed = 299792.458
        self.H0mpc = self.H0/self.cspeed 
        self.rhocrit = (self.h**2)*2.77536627*10**(11)  # Critical density today in M_\odot Mpc^-3
        self.rhomatter = self.omegam*self.rhocrit     

        
        #-------- Define the dark matter halo mass and collapse model

        self.npts=npts#1000#1000#1500#2000 #400       #1000     # Number of points equally spaced in log_10 m to sample. Default = 400
        self.logmmin=logmmin#9.95#8#9.95#8#9.95#8#9.95  #4          # log_10 m/M_\odot of the minimum mass halo to sample. for kmax=1000, should be > 4. Default = 4.
        self.logmmax=logmmax  #17          # log_10 m/M\odit of the maximum mass halo to sample. Default = 17.
        
        #self.npts=1000 #400            # Number of points equally spaced in log_10 m to sample. Default = 400
        #self.logmmin=13.3  #4          # log_10 m/M_\odot of the minimum mass halo to sample. for kmax=1000, should be > 4. Default = 4.
        #self.logmmax=13.5  #17          # log_10 m/M\odit of the maximum mass halo to sample. Default = 17.

        self.c1=0.3222     # Uncomment for Sheth-Torman mass function:
        self.c2=0.707
        self.p=0.3

        #self.c1=0.5     # Uncomment for Press-Schechter mass function:
        #self.c2=1.0
        #self.p=0.0

        self.deltac=1.686                         # Critical density for collapse.           

        #sampling variables
        kmax = 100.
        kmin = 1e-5
        self.k = np.logspace(np.log10(kmin),np.log10(kmax),num=kpoints)
        self.z = [] #filled later
        print("ok")
        self.ukmversion = ukmversion
    #generates self.mthresHOD[z] after z sampling has been set up
    def _setup_mthreshHOD(self,mthreshHOD):
        #Threshold stellar mass for galaxy HOD at all z. Default: 10.5 
        if mthreshHOD.shape[0] == self.z.shape[0]:
            self.mthreshHOD=mthreshHOD
        else:
            #if nthresh is length 1, broadcast over z
            self.mthreshHOD=mthreshHOD[0]*np.ones(self.z.shape[0])
            
    def bias(self,nu):
        y=np.log10(200)
        A=1+0.24*y*np.exp(-(4/y)**4)
        a=0.44*y-0.88
        B=0.183
        b=1.5
        C=0.019+0.107*y+0.19*np.exp(-(4/y)**4)
        c=2.4
    
        return 1-A*nu**a/(nu**a+1.686**a)+B*nu**b+C*nu**c

            
    #set up class variables for the k and z sampling demanded in power spectrum evaluations
    def _setup_k_z_m_grid_functions(self,z,mthreshHOD=mthreshHODdefault,need_ukm=True,need_ykm=False,ukm_version="old"):
        if np.array_equal(self.z,z): #no need to setup the grid functions if they are the same as on the last call
            #now check if the HOD also is unchanged.
            if np.array_equal(self.mthreshHOD,mthreshHOD): #yes, unchanged
                return
            else: #no, changed. set up HOD threshold and return.
                self._setup_mthreshHOD(mthreshHOD)
                return

        #print ("z", z)
        
        # ------- Define the grid in redshift, k and mass.
        
        self.z = z
        self.logm=np.linspace(self.logmmin,self.logmmax,self.npts)             # Mass grid, linear in log_10 m.
        self.lnms=np.log(10**(self.logm))                                      # Mass grid, linear in ln m.

        #--------- set up HOD m_thresh
        self._setup_mthreshHOD(mthreshHOD)

        
        #-------- Calculate cosmology, CAMB power spectra, and variables in the collapse model. (Takes a minute, go get a coffee and come back :).

        # Calculate the linear and non-linear matter power spectrum at each redshift using CAMB. 
        # Units are cubic Mpc as a function of k (e.g. no factors of h).

        self.cambpars = camb.CAMBparams()
        self.cambpars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2)
        print("redoing camb a")
        #self.cambpars.InitPower.set_params(ns=self.ns, r=0)
        self.cambpars.InitPower.set_params(As=self.As*1e-9,ns=self.ns, r=0)
        #self.cambpars.set_for_lmax(2500, lens_potential_accuracy=0);
        self.cambpars.set_matter_power(redshifts=self.z.tolist(), kmax=self.k[-1],k_per_logint=20)
        self.cambresults = camb.get_results(self.cambpars)        
        PK_lin = camb.get_matter_power_interpolator(self.cambpars, nonlinear=False,hubble_units=False, k_hunit=False, kmax=self.k[-1], zmax=self.z[-1])        
        PK_nonlin = camb.get_matter_power_interpolator(self.cambpars, nonlinear=True,hubble_units=False, k_hunit=False, kmax=self.k[-1], zmax=self.z[-1])
        self.pk = PK_lin.P(self.z, self.k, grid=True)
    
        
        self.pknl = PK_nonlin.P(self.z, self.k, grid=True)

        # A few background parameters using CAMB functions
        self.Dist = self.cambresults.comoving_radial_distance(self.z)        # Comoving distance as a function of z.
        self.hubbleconst=self.cambresults.h_of_z(self.z)                     # H(z)     
        self.rhocritz=self.rhocrit*(self.hubbleconst**2)/(self.H0mpc**2)      # Critical density as a function of z. 
        self.omegamz=1./(1.+(1-self.omegam)/(self.omegam*(1+self.z**3)))      # Omega_m as a function of z

        # Get the growth function from CAMB

        growthfn = self.cambresults.get_redshift_evolution(.0001, self.z, ['delta_baryon'])  #Extract the linear growth function from CAMB.
        self.growthfn = growthfn/growthfn[0]

        
        #--------- Define some variables.

        self.nu = np.zeros([self.npts,self.z.shape[0]])              # nu=delta_c^2/sigma^2
        self.mstar = np.zeros([self.z.shape[0]])                     # Mass where nu=1
        self.conc = np.zeros([self.npts,self.z.shape[0]])            # Concentration parameter for NFW profile conc=r_vir/r_s as a function of mass and z.
       # self.conc_200 = np.zeros([self.npts,self.z.shape[0]])            # Concentration parameter for NFW profile conc=r_vir/r_s as a function of mass and z.

        self.rvir3 = np.zeros([self.npts,self.z.shape[0]])           # Virial radius cubed as a function of mass and z
       # self.rvir3_200d = np.zeros([self.npts,self.z.shape[0]])           # Virial radius cubed as a function of mass and z
       # self.rvir3_200c = np.zeros([self.npts,self.z.shape[0]])           # Virial radius cubed as a function of mass and z




        self.conc_virial =  np.zeros([self.npts,self.z.shape[0]]) 
        self.conc_200_c = np.zeros([self.npts,self.z.shape[0]])            # Concentration parameter for NFW profile conc=r_vir/r_s as a function of mass and z.
        self.conc_200_d = np.zeros([self.npts,self.z.shape[0]])            # Concentration parameter for NFW profile conc=r_vir/r_s as a function of mass and z.

        self.rvir3 = np.zeros([self.npts,self.z.shape[0]]) 
        self.r200_3_c =  np.zeros([self.npts,self.z.shape[0]])           # Virial radius cubed as a function of mass and z
        self.r200_3_d =  np.zeros([self.npts,self.z.shape[0]])           # Virial radius cubed as a function of mass and z


        self.sigmam2=np.zeros([self.npts,self.z.shape[0]])           # Variance using top hat window at mass M.
        self.dlogsigmadlogm=np.zeros([self.npts,self.z.shape[0]])    # d\ln\sigma/d\ln m
        self.fsigma=np.zeros([self.npts,self.z.shape[0]])            # Collapse fraction assuming Press-Schechter or Sheth-Torman.
        self.nfn=np.zeros([self.npts,self.z.shape[0]])               # Differential number density of halos n(m,z). Note, this is comoving number density.
        self.halobias=np.zeros([self.npts,self.z.shape[0]])          # Linear halo bias

        # Loops to calculate variables above.

        # Redshift loop.
        print("starting redshift loop")
        
        t1=time.time()
        omega = self.omegam*(1+self.z)**3/(self.omegam*(1+self.z)**3+(1-self.omegam))

        self.deltav = 18*np.pi**2+82*(omega - 1)-39*(omega -1)**2
        for j in range(0,self.z.shape[0]):
            Rs=(3.*10**(self.logm)/(4.*np.pi*self.rhomatter))**(1./3)
            # Mass loop
           

            for i in range(0,self.logm.shape[0]):
                # Argument of the window functions. Note, Rs and k are both comoving.
                xs=Rs[i]*self.k
                # Define the window function at different masses. For numerical accuracy, use Taylor expansion at low kR.
                self.window = np.piecewise(xs,[xs>.01,xs<=.01],[lambda xs: (3./(xs**3))*(np.sin(xs)-(xs)*np.cos(xs)), lambda xs: 1.-xs**2./10.])
                # \sigma^2 as a function of mass, implement redshift dependence through the growth function.
                self.sigmam2[i,j] = (self.growthfn[j]**2)*np.trapz(self.k*self.k*self.pk[0,:]*self.window**2/(2.*np.pi**2),self.k)
                self.nu[i,j] = self.deltac**2/self.sigmam2[i,j]
                # Derivative of the window function, use Taylor expansion at low kR.
                self.dwindow2dm = -np.piecewise(xs,[xs>.01,xs<=.01],[lambda xs: (6./xs**6)*(np.sin(xs)-xs*np.cos(xs))*(np.sin(xs)*(-3.+xs**2) + 3.*xs*np.cos(xs)), lambda xs: -2.*xs**2./15.])
                # Calculate log derivative of \sigma; note that powers of the growth function cancel so no redshift dependence.
                self.dlogsigmadlogm[i,j] = (1./self.sigmam2[i,0])*np.trapz(self.k*self.k*self.pk[0,:]*self.dwindow2dm/(4.*np.pi**2),self.k)
                # Collapse fraction assuming Press-Schechter or Sheth-Torman.
                self.fsigma[i,j] = self.c1*np.sqrt(2.*self.c2/np.pi)*(1.+(self.sigmam2[i,j]/(self.c2*self.deltac**2))**self.p)*(self.deltac/np.sqrt(self.sigmam2[i,j]))*np.exp(-0.5*self.c2*self.deltac**2/self.sigmam2[i,j])
                # Put together pieces of the differential number density and halo bias.
                self.nfn[i,j]=(self.rhomatter/(10**(2.*self.logm[i])))*self.fsigma[i,j]*self.dlogsigmadlogm[i,j]
                self.halobias[i,j]=1.+(self.c2*(self.deltac**2/self.sigmam2[i,j])-1.)/self.deltac+(2.*self.p/self.deltac)/(1.+ (self.c2*(self.deltac**2/self.sigmam2[i,j]))**self.p )
            # Calculate the NFW concentration parameter and halo virial radius.
            
            
            
            
            
            
            
            
            
           # self.deltav = 178*self.omegamz[j]**(0.45)                  # Virial overdensity, approximated as in Eke et al. 1998.        
            self.mstar[j] = self.logm[(np.abs(self.nu[:,j]-1.)).argmin()]
            # Several choices for the NFW halo concentration parameter as a function of mass. Here, assumed deterministic.
            #conc[:,j] = 9.*(10**(-.13*(logm-mstar[j]*np.ones(logm.shape[0]))))/(1+z[j])    # Bullock model from Cooray/Sheth            
           # self.conc_200[:,j] = 5.71*(( self.h*0.5*10**(self.logm-12) )**(-0.084))*(1+self.z[j])**(-0.47) # Duffy08 all model from 1005.0411

            self.conc_virial[:,j] = 7.85*(( self.h*0.5*10**(self.logm-12) )**(-0.081))*(1+self.z[j])**(-0.71) # Duffy08 all model from 1005.0411
            # Cube of the virial radius. Note, this is a physical distance, not a comoving distance.
            self.rvir3[:,j] = 3.*(10**self.logm)/(4.*np.pi*self.deltav[j]*self.rhocritz[j])
            
            self.conc_200_c[:,j] = 5.71*(( self.h*0.5*10**(self.logm-12) )**(-0.084))*(1+self.z[j])**(-0.47) # Duffy08 all model from 1005.0411
            
            self.conc_200_d[:,j] =  10.14*(( self.h*0.5*10**(self.logm-12) )**(-0.081))*(1+self.z[j])**(-1.01) # Duffy08 all model from 1005.0411

            self.r200_3_c[:,j] = 3.*(10**self.logm)/(4.*np.pi*200*self.rhocritz[j])
            self.r200_3_d[:,j] = 3.*(10**self.logm)/(4.*np.pi*200*self.rhomatter*(1+self.z[j])**3)

             
        mhalos=np.exp(self.lnms)
        zslessthan3=self.z.copy()
        zslessthan3[zslessthan3>3]=3
        beta0 = 0.589
        gamma0 = 0.864
        phi0 = -0.729
        eta0 = -0.243
        beta  = beta0  * (1+zslessthan3)**(0.20)
        phi   = phi0   * (1+zslessthan3)**(-0.08)
        eta   = eta0   * (1+zslessthan3)**(0.27)
        gamma = gamma0 * (1+zslessthan3)**(-0.01)
        izs,ialphas = np.loadtxt("./alpha_consistency.txt",unpack=True) # FIXME: hardcoded
        alpha = interp1d(izs,ialphas,bounds_error=True)(zslessthan3)
        nu=np.sqrt(self.nu)
        self.fsigma=alpha*np.sqrt(self.nu)*(1. + (beta*nu)**(-2.*phi))*(nu**(2*eta))*np.exp(-gamma*nu**2./2.)#A*((np.sqrt(self.sigmam2)/b)**-a+1)*np.exp(-c/self.sigmam2)# self.c1*np.sqrt(2.*self.c2/np.pi)*(1.+(self.sigmam2[i,j]/(self.c2*self.deltac**2))**self.p)*(self.deltac/np.sqrt(self.sigmam2[i,j]))*np.exp(-0.5*self.c2*self.deltac**2/self.sigmam2[i,j])
        dndm=self.rhomatter/mhalos[:,np.newaxis]**2*self.fsigma*self.dlogsigmadlogm
        self.mhalos = np.exp(self.lnms)
        
      #  self.m200d = np.zeros([self.npts,self.z.shape[0]])   
       # self.r200d = np.zeros([self.npts,self.z.shape[0]])  
        
       # for zindex in range(0,self.z.shape[0]):
       #     for mindex in range(0,self.npts):
       #         m,r = self.mrvir_to_mrdelta_mean(200,zindex,self.conc[mindex,zindex],self.rvir3[mindex,zindex]**(1/3),self.mhalos[mindex])   #we need this to use the tinker halo function which is defined for m _200(mean) not m_virial or m_200(crit)
       #         self.m200d[mindex,zindex] = m 
       #         self.r200d[mindex,zindex] = r
        
 #       dn200dm[1:] = (self.m200d[1:]- self.m200d[:-1])/(self.mhalos[1:,np.newaxis]-self.mhalos[:-1,np.newaxis])
#        #dn200dm = np.zeros(self.nfn.shape)
#        dn200dm[0] = dn200dm[1]
        
       # self.dn200dm = dn200dm
       # self.nfn = dndm*dn200dm
        if self.mdef == "200_mean":
            self.halobias=self.bias(np.sqrt(self.nu)) #this will not work for 
            
            
            self.mvir = np.zeros([self.npts,self.z.shape[0]])
            self.rvir = np.zeros([self.npts,self.z.shape[0]])
            self.m200c = np.zeros([self.npts,self.z.shape[0]])
            self.r200c = np.zeros([self.npts,self.z.shape[0]])
            
            for zindex in range(0,len(self.z)):
                for mindex in range(0,len(self.mhalos)):
     
                    
                    m,r = self.m_to_mvir(zindex,self.r200_3_d[mindex,zindex]**(1/3),self.mhalos[mindex],200*self.rhomatter*(1+self.z[zindex])**3)
                    self.mvir[mindex,zindex] = m
                    self.rvir[mindex,zindex] = r
                   # m,r = self.mdeltad_to_mdeltac(zindex,self.r200_3_d[mindex,zindex]**(1/3),self.mhalos[mindex],200*self.rhomatter*(1+self.z[zindex])**3)
                        
                   # self.m200c[mindex,zindex] = m
                   # self.r200c[mindex,zindex] = r
                    
                    m,r = self.mrvir_to_mrdelta(200,zindex,self.conc_virial[mindex,zindex],self.rvir[mindex,zindex],self.mvir[mindex,zindex])
                        
                    self.m200c[mindex,zindex] = m
                    self.r200c[mindex,zindex] = r
        #self.halobias=self.bias(np.sqrt(self.nu))
    
    
        print("done in",time.time()-t1,"seconds")

        #----- Properties of the dark matter halo.

        # Calculate the normalized FT of the NFW profile as a function of halo mass and z. Note, the FT is in terms of the 
        # comoving wavenumber.
        if (need_ukm==False and need_ykm==False):
            return
        if need_ukm:
            self.ukm = np.zeros([self.npts,self.z.shape[0],self.k.shape[0]]) 
            
            print("starting ukm loop")
            t1=time.time()
            
            
            
            for j in range(0,self.z.shape[0]):
                for i in range(0,self.logm.shape[0]):
                    c = self.conc_200_d[i,j]
                    mc = np.log(1+c)-c/(1.+c)
                    rs = (self.r200_3_d[i,j]**(0.33333333))/c
                    # Include a factor of (1+z) to account for physical k vs comoving k.
                    x = self.k*rs*(1+self.z[j])
                    Si, Ci = scipy.special.sici(x)
                    Sic, Cic = scipy.special.sici((1.+c)*x)
                    self.ukm[i,j,:] = (np.sin(x)*(Sic-Si) - np.sin(c*x)/((1+c)*x) + np.cos(x)*(Cic-Ci))/mc
            print("done in",time.time()-t1,"seconds")
            #----- Load the precomputed normalized FT's of the gas profiles as a function of mass, z, and k (comoving). interpolate to the present k and z sampling.
            
            if self.ukmversion=="old":
                
                profilesuffix = '_hires_fixed' 
                ukgas_universal_precomp = np.load('gasprofile_universal'+profilesuffix+'.npy')
                ukgas_AGN_precomp = np.load('gasprofile_AGN'+profilesuffix+'.npy')
                ukgas_SH_precomp = np.load('gasprofile_SH'+profilesuffix+'.npy')
                npzfile = np.load('gasprofile_sampling'+profilesuffix+'.npz')
                m_precomp = npzfile['logm_precomp']
                z_precomp = npzfile['z_precomp']
                k_precomp  = npzfile['k_precomp']
                #print ("test",ukgas_AGN_precomp.shape,m_precomp.shape,z_precomp.shape,k_precomp.shape)
                #interpolate these to the z and k precomps that we need. mass precomp is assumed to be fixed.
                self.ukgas_universal = np.zeros( (self.npts, self.z.shape[0], self.k.shape[0]) )
                self.ukgas_AGN = np.zeros( (self.npts, self.z.shape[0], self.k.shape[0]) )
                self.ukgas_SH = np.zeros( (self.npts, self.z.shape[0], self.k.shape[0]) )
                for m_id in range(m_precomp.shape[0]):
                    interp = interp2d(k_precomp,z_precomp,ukgas_universal_precomp[m_id],bounds_error=False,fill_value=0)
                    self.ukgas_universal[m_id] = interp(self.k,self.z)
                    interp = interp2d(k_precomp,z_precomp,ukgas_AGN_precomp[m_id],bounds_error=False,fill_value=0)
                    self.ukgas_AGN[m_id] = interp(self.k,self.z)            
                    interp = interp2d(k_precomp,z_precomp,ukgas_SH_precomp[m_id],bounds_error=False,fill_value=0)
                    self.ukgas_SH[m_id] = interp(self.k,self.z)
                assert m_precomp.shape[0] == self.npts
                print(m_precomp[0],self.logmmin)
                assert m_precomp[0] == self.logmmin
                assert m_precomp[-1] == self.logmmax 
            elif self.ukmversion=="new":
                
                print("getting new ukm")
                profilesuffix = '_hires_fixed' 
                ukgas_AGN_precomp = np.load('./fiona_gasprofiles/gasprofile_AGN.npy')
                npzfile = np.load('./fiona_gasprofiles/gasprofile_sampling.npz')
                m_precomp = npzfile['logm_precomp']
                z_precomp = npzfile['z_precomp']
                k_precomp  = npzfile['k_precomp']
                #print ("test",ukgas_AGN_precomp.shape,m_precomp.shape,z_precomp.shape,k_precomp.shape)
                #interpolate these to the z and k precomps that we need. mass precomp is assumed to be fixed.
                self.ukgas_AGN = np.zeros( (self.npts, self.z.shape[0], self.k.shape[0]) )
                for m_id in range(m_precomp.shape[0]):
                    interp = interp2d(k_precomp,z_precomp,ukgas_AGN_precomp[m_id],bounds_error=False,fill_value=0)
                    self.ukgas_AGN[m_id] = interp(self.k,self.z)            
                assert m_precomp.shape[0] == self.npts
                print(m_precomp[0],self.logmmin)
                assert m_precomp[0] == self.logmmin
                assert m_precomp[-1] == self.logmmax 
            print("ukm set up")
        if need_ykm:
            #p#rint()
             #   if kszpsz_config.precompute_ytsz==False:
              #      
                    yprofilesampling = np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/master_secondaries/Code/yk_battaglia_sampling_001_6.npz")
    
                    kprecomp=yprofilesampling["k_precomp"]
                    mprecomp=yprofilesampling["mhalos_precomp"]
                    zprecomp=yprofilesampling["zs"]
                    
                    yprofile=np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/master_secondaries/Code/yk_battaglia_001_6.npy")
                    print(yprofile.shape)
                    ykprecomp=yprofile
                     
                    assert mprecomp[0] == np.exp(self.lnms[0])
                    assert mprecomp[-1] == np.exp(self.lnms[-1]) 
                    
                    
                    self.ykm=np.zeros((self.npts,self.z.shape[0],self.k.shape[0]))
                    for mind in range(0,len(self.lnms)):
                        #for zind in range(0,len(self.z)):
                         #   for kind in range(0,len(self.k)):
                                
                         yk=interp2d(zprecomp,kprecomp,ykprecomp[:,mind,:])
                         self.ykm[mind]=np.transpose(yk(self.z,self.k))
                    print("ykm set up")
                        


     
            
    # Function to compute dark matter density profile in units of M_\odot Mpc^-3, specifying
    # c = concentration
    # rv = virial radius
    # mv = mass
    # rvals = values of r to return profile at. 
    # Note: All distances are physical distances, not comoving distances.
    def dmprofile(self,c,rv,mv,rvals):
        xvals=c*rvals/rv
        nfw=1./(xvals*(1+xvals)**2)
        mc=np.log(1.+c ) - c/(1.+c)
        rhos=(c**3)*mv/(4.*mc*np.pi*rv**3)
        return rhos*nfw

    
    # Function to convert r_vir and m_vir to r_delta and m_delta.precompute_ytsz
    # delta = overdensity (e.g. 200, 500, etc)
    # zindex = redshift index to do the computation
    # c = concentration
    # rv = virial radius
    # mv = virial mass
    # Note: All distances are physical distances, not comoving distances.
    def mrvir_to_mrdelta(self,delta,zindex,c,rv,mv):
        rdeltavals = np.linspace(rv/10,2*rv,1000)
        mc=np.log(1.+c ) - c/(1.+c)
        mcr=np.log(1.+ (c*rdeltavals/rv) ) - (c*rdeltavals/rv)/(1.+(c*rdeltavals/rv))
        fn = np.abs(rdeltavals**3 - 3.*mv*mcr/(4.*np.pi*delta*self.rhocritz[zindex]*mc))
        rdelta = rdeltavals[fn.argmin()]
        mcrdelta = np.log(1.+ (c*rdelta/rv) ) - (c*rdelta/rv)/(1.+(c*rdelta/rv))
        mdelta = mcrdelta*mv/mc
        return mdelta,rdelta
    
    def mrvir_to_mrdelta_mean(self,delta,zindex,c,rv,mv):   #we need this to use the tinker halo function which is defined for m _200(mean) not m_virial or m_200(crit)
        rdeltavals = np.linspace(rv/10,2*rv,1000)
        mc=np.log(1.+c ) - c/(1.+c)
        mcr=np.log(1.+ (c*rdeltavals/rv) ) - (c*rdeltavals/rv)/(1.+(c*rdeltavals/rv))
        fn = np.abs(rdeltavals**3 - 3.*mv*mcr/(4.*np.pi*delta*self.rhomatter*(1+self.z[zindex])**3*mc))
        rdelta = rdeltavals[fn.argmin()]
        mcrdelta = np.log(1.+ (c*rdelta/rv) ) - (c*rdelta/rv)/(1.+(c*rdelta/rv))
        mdelta = mcrdelta*mv/mc
        return mdelta,rdelta
    
    def m_to_mvir(self,zindex,r_old,m_old,old_density_definition):
        rvirvals = np.linspace(r_old/10,2*r_old,10000)
        mvirvals = 4*np.pi*rvirvals**3*self.deltav[zindex]*self.rhocritz[zindex]/3
        
        
        concs = 7.85*(( self.h*0.5*10**(np.log10(mvirvals)-12) )**(-0.081))*(1+self.z[zindex])**(-0.71) # Duffy08 all model from 1005.0411

        mc=np.log(1.+concs ) - concs/(1.+concs)
        
        mcr=np.log(1.+ (concs*r_old/rvirvals) ) - (concs*r_old/rvirvals)/(1.+(concs*r_old/rvirvals))
        fn = np.abs(r_old**3 - 3.*mvirvals*mcr/(4.*np.pi*old_density_definition*mc))
        rvir = rvirvals[fn.argmin()]
        mvir = mvirvals[fn.argmin()]
        return mvir,rvir
    
    def mdeltad_to_mdeltac(self,zindex,r_old,m_old,old_density_definition):
        rcvals = np.linspace(r_old/10,2*r_old,10000)
        
        mcvals = 4*np.pi/3*rcvals**3* 200 * self.rhocritz[zindex]
        
        concs = 5.71*(( self.h*0.5*10**(mcvals-12) )**(-0.084))*(1+self.z[zindex])**(-0.47) # Duffy08 all model from 1005.0411
            

        mc=np.log(1.+concs ) - concs/(1.+concs)
        
        mcr=np.log(1.+ (concs*r_old/rcvals) ) - (concs*r_old/rcvals)/(1.+(concs*r_old/rcvals))
        fn = np.abs(r_old**3 - 3.*mcvals*mcr/(4.*np.pi*old_density_definition*mc))
        rc = rcvals[fn.argmin()]
        mc = mcvals[fn.argmin()]
        return mc,rc

    
    #-------- HOD following the "SIG_MOD1" HOD model of 1104.0928 and 1103.2077 with redshift dependence from 1001.0015.
    # See also 1512.03050, where what we are using corresponds to their "Baseline HOD" model.

    # Function to compute the stellar mass Mstellar from a halo mass mv at redshift z.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    def Mstellar_halo(self,z,log10mhalo):
        a = 1./(1+z)
        if (z<=0.8):
            Mstar00=10.72
            Mstara=0.55
            M1=12.35
            M1a=0.28
            beta0=0.44
            beta_a=0.18
            gamma0=1.56
            gamma_a=2.51
            delta0=0.57
            delta_a=0.17
        if (z>0.8):
            Mstar00=11.09
            Mstara=0.56
            M1=12.27
            M1a=-0.84
            beta0=0.65
            beta_a=0.31
            gamma0=1.12
            gamma_a=-0.53
            delta0=0.56
            delta_a=-0.12            
        log10M1 = M1 + M1a*(a-1)
        log10Mstar0 = Mstar00 + Mstara*(a-1)
        beta = beta0 + beta_a*(a-1)
        gamma = gamma0 + gamma_a*(a-1)
        delta = delta0 + delta_a*(a-1)
        log10mstar = np.linspace(-18,18,1000)
        mh = -0.5 + log10M1 + beta*(log10mstar-log10Mstar0) + 10**(delta*(log10mstar-log10Mstar0))/(1.+ 10**(-gamma*(log10mstar-log10Mstar0)))
        mstar = np.interp(log10mhalo,mh,log10mstar)
        return mstar

    
    # Function to compute halo mass as a function of the stellar mass.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    def Mhalo_stellar(self,z,log10mstellar):
        a = 1./(1+z) 
        if (z<=0.8):
            Mstar00=10.72
            Mstara=0.55
            M1=12.35
            M1a=0.28
            beta0=0.44
            beta_a=0.18
            gamma0=1.56
            gamma_a=2.51
            delta0=0.57
            delta_a=0.17
        if (z>0.8):
            Mstar00=11.09
            Mstara=0.56
            M1=12.27
            M1a=-0.84
            beta0=0.65
            beta_a=0.31
            gamma0=1.12
            gamma_a=-0.53
            delta0=0.56
            delta_a=-0.12            
        log10M1 = M1 + M1a*(a-1)
        log10Mstar0 = Mstar00 + Mstara*(a-1)
        beta = beta0 + beta_a*(a-1)
        gamma = gamma0 + gamma_a*(a-1)
        delta = delta0 + delta_a*(a-1)
        log10mstar = log10mstellar
        log10mh = -0.5 + log10M1 + beta*(log10mstar-log10Mstar0) + 10**(delta*(log10mstar-log10Mstar0))/(1.+ 10**(-gamma*(log10mstar-log10Mstar0)))
        return log10mh

    
    # Number of central galaxies as a function of halo mass and redshift.
    # lnms = natural log of halo masses
    # z = redshifts
    # log10Mst_thresh = log10 of the stellar mass threshold. Defined above as mthresh with the other free parameters.
    # returns array[mass,z]
    def Ncentral(self,lnms,z,log10Mst_thresh):
        logm10=np.log10(np.exp(lnms))
        log10Mst=self.Mstellar_halo(z,logm10) 
        sigmalogmstar=0.2
        log10Mst_threshar=log10Mst_thresh*np.ones(log10Mst.shape[0])
        arg = (log10Mst_threshar-log10Mst)/(np.sqrt(2)*sigmalogmstar)
        return 0.5-0.5*scipy.special.erf(arg)

    
    # Number of satellite galaxies.
    # lnms = natural log of halo masses
    # z = redshifts
    # log10Mst_thresh = log10 of the stellar mass threshold. Defined above as mthresh with the other free parameters.
    # returns array[mass,z]
    def Nsatellite(self,lnms,z,log10Mst_thresh):
        logm10=np.log10(np.exp(lnms))
        Bsat=9.04
        betasat=0.74
        alphasat=1.
        Bcut=1.65
        betacut=0.59
        Msat=(10.**(12.))*Bsat*10**((self.Mhalo_stellar(z,log10Mst_thresh)-12)*betasat)
        Mcut=(10.**(12.))*Bcut*10**((self.Mhalo_stellar(z,log10Mst_thresh)-12)*betacut)
        return self.Ncentral(lnms,z,log10Mst_thresh)*((np.exp(lnms)/Msat)**alphasat)*np.exp(-Mcut/(np.exp(lnms)))

    ####---- functions for P_ge where e is all mass
    # Number of centrals in a mass bin
    # returns array[mass,z]
    def Ncentral_binned(self,lnms,z,log10Mst_thresh,log10mlow,log10mhigh):
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        logm10=np.log10(np.exp(lnms))
        log10Mst=self.Mstellar_halo(z,logm10)
        sigmalogmstar=0.2
        log10Mst_threshar=log10Mst_thresh*np.ones(log10Mst.shape[0])
        arg = (log10Mst_threshar-log10Mst)/(np.sqrt(2)*sigmalogmstar)
        Ncentral=np.zeros(lnms.shape[0])
        Ncentral[(lnms>=lnmlow)&(lnms<=lnmhigh)]=0.5-0.5*scipy.special.erf(arg[(lnms>=lnmlow)&(lnms<=lnmhigh)])
        #print (Ncentral)
        return Ncentral
    
    # Number of satellites in a mass bin
    # returns array[mass,z]                 #   no, it returns an array[mass] at a specific z... F
    def Nsatellite_binned(self,lnms,z,log10Mst_thresh,log10mlow,log10mhigh):
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        logm10=np.log10(np.exp(lnms))
        Bsat=9.04
        betasat=0.74
        alphasat=1.
        Bcut=1.65
        betacut=0.59
        Msat=(10.**(12.))*Bsat*10**((self.Mhalo_stellar(z,log10Mst_thresh)-12)*betasat)
        Mcut=(10.**(12.))*Bcut*10**((self.Mhalo_stellar(z,log10Mst_thresh)-12)*betacut)
        #   the below is eq 12 of 1103.2077
        Nsatellite = self.Ncentral(lnms,z,log10Mst_thresh)*((np.exp(lnms)/Msat)**alphasat)*np.exp(-Mcut/(np.exp(lnms)))
        Nsatellite[(lnms<=lnmlow)] = 0.0
        Nsatellite[(lnms>=lnmhigh)] = 0.0
        #print (Nsatellite)
        return Nsatellite
    ####----
    
    # Average comoving number density of gsatflux_interpolatedalaxies.
    # lnms = natural log of halo masses
    # z = redshifts
    # nfn = halo mass function
    # Ngal = number of galaxies as a function of halo mass and redshift
    def Nbargal(self,lnms,z,nfn,Ngal):
        nbargal=np.zeros([z.shape[0]])
        for j in range(0,z.shape[0]):
            nbargal[j]=np.trapz(np.exp(lnms)*nfn[:,j]*Ngal[:,j],lnms)
        return nbargal

    
    # Function to calculate mass-binned galaxy bias.
    def _galaxybias(self,lnms,z,nfn,Ngalc,Ngals,halobias,log10mlow,log10mhigh):
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        lnmsin=lnms[(lnms>=lnmlow)&(lnms<=lnmhigh)]
        nfnin=nfn[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalcin=Ngalc[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalsin=Ngals[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        halobiasin=halobias[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        nbargaltot=self.Nbargal(lnmsin,z,nfnin,Ngalcin+Ngalsin)
        bgal = np.zeros([z.shape[0]])
        for j in range(0,z.shape[0]):
            bgal[j]=np.trapz((np.exp(lnmsin))*nfnin[:,j]*(Ngalcin[:,j]+Ngalsin[:,j])*halobiasin[:,j]/nbargaltot[j],lnmsin)

        return bgal

    
    # Function to calculate mass-binned galaxy number density.
    def Nbargal_binned(self,lnms,z,nfn,Ngal,log10mlow,log10mhigh):
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        lnmsin=lnms[(lnms>=lnmlow)&(lnms<=lnmhigh)]
        nfnin=nfn[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalin=Ngal[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        #print (lnmsin.shape[0])
        nbargalbinned=np.zeros(z.shape[0])
        for j in range(0,z.shape[0]):
            nbargalbinned[j]=np.trapz(np.exp(lnmsin)*nfnin[:,j]*Ngalin[:,j],lnmsin)
        return nbargalbinned




    #------- Properties of the gas profile within halos. 

    #used to select gas profile in power spectra
    def _get_gas_profile(self,gasprofile):
        if gasprofile == 'universal':
            return self.ukgas_universal
        if gasprofile == 'AGN':
            return self.ukgas_AGN
        if gasprofile == 'SH':
            return self.ukgas_SH
        
    
    # Function to compute gas density profile from Komatsu and Seljak (astro-ph/0106151) in units of M_\odot Mpc^-3, specifying
    # c = concentration
    # rv = virial radius of dark matter halo
    # mv = mass of dark matter halo
    # rvals = values of r to return profile at (in units of Mpc). 
    # rstar = value at which slope of gas profile matches that of NFW. Choose to be within a factor of 2 larger or smaller than rv, and no effect on profile.
    # Note: All distances are physical distances, not comoving distances.
    def gasprofile(self,c,rv,mv,rvals,rstar):
        xstar=c*rstar/rv
        xvals=c*rvals/rv
        nfw=1./(xvals*(1+xvals)**2)
        mc=np.log(1.+c ) - c/(1.+c)
        rhos=(c**3)*mv/(4.*mc*np.pi*rv**3)
        gamma=1.15+0.01*(c-6.5)
        indexxstar=(np.abs(xvals - xstar)).argmin()
        sstar=-(1.+2.*xstar/(1.+xstar))
        mxstar=np.log(1.+xstar) - xstar/(1.+xstar)
        mxstarint = 1. - np.log(1.+xstar)/xstar
        eta0=-(3./(gamma*sstar))*(c*mxstar/(xstar*mc)) + 3.*(gamma-1)*c*mxstarint/(gamma*mc)
        gas = ((1. - (3./eta0)*((gamma-1.)/gamma)*(c/mc)*(1.-np.log(xvals+1.)/(xvals)))**(1./(gamma-1.)))
        gasrescale=(nfw[indexxstar]/gas[indexxstar])*rhos*(self.ombh2/self.omch2)
        return np.piecewise(xvals,[xvals<=xstar,xvals>xstar],[lambda xvals: gasrescale*((1. - (3./eta0)*((gamma-1.)/gamma)*(c/mc)*(1.-np.log(xvals+1.)/(xvals)))**(1./(gamma-1.))), lambda xvals: rhos*(self.ombh2/self.omch2)/(xvals*(1+xvals)**2)])


    # Function to compute gas density profile from Battaglia (1607.02442) AGN feedback model in units of M_\odot Mpc^-3, specifying
    # c = concentration
    # rv = virial radius of dark matter halo
    # mv = mass of dark matter halo
    # rvals = values of r to return profile at (in units of Mpc). 
    # zindex = index specifying the redshift 
    # Note: All distances are physical distances, not comoving distances.
    def gasprofile_battaglia_AGN(self,rvals,zindex,mind):    
        m200, r200 = self.m200c[mind,zindex],self.r200c[mind,zindex]#self.mrvir_to_mrdelta(200,zindex,c,rv,mv)
        rho0 = 4000.*((m200/(10**(14)))**(0.29))*(1+self.z[zindex])**(-0.66)
        alpha = 0.88*((m200/(10**(14)))**(-0.03))*(1+self.z[zindex])**(0.19)
        beta = 3.83*((m200/(10**(14)))**(0.04))*(1+self.z[zindex])**(-0.025)
        xc = 0.5
        gamma = -0.2
        xxs = rvals/r200 
        ans = self.rhocritz[zindex]*rho0*((xxs/xc)**gamma)*( 1. +(xxs/xc)**alpha )**(-(beta+gamma)/alpha)
        return ans


    # Function to compute gas density profile from Battaglia (1607.02442) Shock Heating feedback model in units of M_\odot Mpc^-3, specifying
    # c = concentration
    # rv = virial radius of dark matter halo
    # mv = mass of dark matter halo
    # rvals = values of r to return profile at (in units of Mpc). 
    # zindex = index specifying the redshift 
    # Note: All distances are physical distances, not comoving distances.
    def gasprofile_battaglia_SH(self,c,rv,mv,rvals,zindex):    
        m200, r200 = self.mrvir_to_mrdelta(200,zindex,c,rv,mv)
        rho0 = 19000.*((m200/(10**(14)))**(0.09))*(1+self.z[zindex])**(-0.95)
        alpha = 0.70*((m200/(10**(14)))**(-0.017))*(1+self.z[zindex])**(0.27)
        beta = 4.43*((m200/(10**(14)))**(0.005))*(1+self.z[zindex])**(0.037)
        xc = 0.5
        gamma = -0.2
        xxs = rvals/r200
        ans = self.rhocritz[zindex]*rho0*((xxs/xc)**gamma)*( 1. +(xxs/xc)**alpha )**(-(beta+gamma)/alpha)
        return ans

   # def calc_yk_profiles():
        
        

    #------ Calculate the normalized FT of gas density profile as a function of mass and z
    def calc_gasprofiles(self,gasprofile,prefix): #TODO (MM)
        #!!!!!!!This takes up to 30 minutes, for convenience I've precomputed this in the files 'gasprofile_universal.npy' 
        
        # 'gasprofile_AGN.npy' and 'gasprofile_SH.npy' for the parameters:            
        nzeds=self.z.shape[0] 
        nks=self.k.shape[0]           
      
        ukout = np.zeros([self.npts,nzeds,nks])
        
        if gasprofile in ['universal']:
            
        
            for j in range(0,self.z.shape[0]):
                for i in range(0,self.logm.shape[0]):
                    c=self.conc[i,j]
                    rv=(self.rvir[i,j])#**(0.3333333)
                    rvals=np.linspace(0.0001,rv,10000)
                    rstar=rv
                    mv=self.rvir[i,j]#10**self.logm[i]
                    mgas=4.*np.pi*np.trapz((rvals**2)*self.gasprofile(c,rv,mv,rvals,rstar),rvals)
                    for q in range(0,self.k.shape[0]):
                        kphys=self.k[q]*(1+self.z[j])
                        ukout[i,j,q] = np.trapz(4.*np.pi*(rvals**2)*(np.sin(kphys*rvals)/(kphys*rvals))*self.gasprofile(c,rv,mv,rvals,rstar)/mgas,rvals)
        
            np.save(prefix+'gasprofile_universal', ukout)

        if gasprofile in ['AGN']:
            
            for j in range(0,self.z.shape[0]):
                print(j)
                for i in range(0,self.logm.shape[0]):
                    print(i)
                    c=self.conc[i,j]
                    rv=(self.rvir[i,j])#**(0.3333333)
                    rvals=np.linspace(0.0001,rv,1000)
                    rstar=rv
                    mv=self.rvir[i,j]#10**self.logm[i]
                    mgas_AGN = 4.*np.pi*np.trapz((rvals**2)*self.gasprofile_battaglia_AGN(rvals,j,i),rvals)
                    for q in range(0,self.k.shape[0]):
                        kphys=self.k[q]*(1+self.z[j])
                        ukout[i,j,q] = np.trapz(4.*np.pi*(rvals**2)*(np.sin(kphys*rvals)/(kphys*rvals))*self.gasprofile_battaglia_AGN(c,rv,mv,rvals,j,i)/mgas_AGN,rvals)
        
            np.save(prefix+'gasprofile_AGN', ukout)

            
        if gasprofile in ['SH']:
            
            for j in range(0,self.z.shape[0]):
                for i in range(0,self.logm.shape[0]):
                    c=self.conc[i,j]
                    rv=(self.rvir3[i,j])**(0.3333333)
                    rvals=np.linspace(0.0001,rv,10000)
                    rstar=rv
                    mv=10**self.logm[i]
                    mgas_SH = 4.*np.pi*np.trapz((rvals**2)*self.gasprofile_battaglia_SH(c,rv,mv,rvals,j),rvals)
                    for q in range(0,self.k.shape[0]):
                        kphys=self.k[q]*(1+self.z[j])
                        ukout[i,j,q] = np.trapz(4.*np.pi*(rvals**2)*(np.sin(kphys*rvals)/(kphys*rvals))*self.gasprofile_battaglia_SH(c,rv,mv,rvals,j)/mgas_SH,rvals)
        
            np.save(prefix+'gasprofile_SH', ukout)


        np.savez(prefix+'gasprofile_sampling.npz',logm_precomp=self.logm,z_precomp=self.z,k_precomp=self.k)
        
    def flux_cent_interpolated(self,frequency): #Fiona
        
        '''
        
        I want this to give a two-dimensional array of the spectral flux density of central galaxies
        running over [i,j]=[mass,redshift].
        
        For now I will interpolate a precomputed version, because S_nu is a function of stellar mass
        and my halo mass-> galactic mass function takes a long time (the function in halomodel.py)
        is not adequate as is breaks down at z>~1.
        
        '''
      #  intensities_precomputed = np.load('central_intensities_'+str(frequency)+'_morepoints.npy')
        intensities_precomputed = np.load('flux_cent_'+str(frequency)+'_new_1000k.npy')
        
        npzfile = np.load('new_sampling_100_points.npz')
        m_precomp = npzfile['logm_precomp']
        z_precomp = npzfile['z_precomp']
        #need more low-z points.
        
        m_precomp=np.log10(np.exp(m_precomp))

        intensity_array=np.zeros((self.npts,self.z.shape[0]))
        for m_id in range(m_precomp.shape[0]):
            interp = interp1d(z_precomp,intensities_precomputed[m_id],bounds_error=False,fill_value=0)
            intensity_array[m_id]=interp(self.z)
            
        assert m_precomp.shape[0] == self.npts
        assert float(m_precomp[0]) == float(self.logmmin)
        assert float(m_precomp[-1]) == float(self.logmmax )
        
        return intensity_array
   
        

    def satflux_interpolated(self,frequency): #Fiona
        '''
        I want this to read in the precomputed .npy files
        sat_flux_353.npy
        sat_flux_545.npy
        sat_flux_857.npy
        to get the flux from the satellite galaxies at the mass values of 
        '''
        sat_flux_precomputed = np.load('satflux_'+str(frequency)+'_moresampling.npy')
        sat_flux_precomputed=np.load('satflux_353_moresampling_1000k.npy')
        npzfile = np.load('new_sampling_100_points.npz')
        m_precomp = npzfile['logm_precomp']
        z_precomp = npzfile['z_precomp']
        
        m_precomp=np.log10(np.exp(m_precomp))
    
        satflux_array=np.zeros((self.npts,self.z.shape[0]))
    
        for m_id in range(m_precomp.shape[0]):
            interp = interp1d(z_precomp,sat_flux_precomputed[m_id],bounds_error=False,fill_value=0)
            satflux_array[m_id]=interp(self.z)
   

        
      
        assert m_precomp.shape[0] == self.npts
       
        assert float(m_precomp[0]) == float(self.logmmin)
        assert float(m_precomp[-1]) == float(self.logmmax) 
        
        return satflux_array

    # Function to calculate the one-halo auto and cross power spectrum for matter, gas, galaxies.
    # z =  redshifts
    # lnms = natural log of halo masses
    # k = comoving wavenumbers
    # uk1 = FT of galaxy run (if galaxy part of correlation), otherwise mass or gas
    # uk2 = FT of mass or gas.
    # nfn = halo mass function
    # Ngalc = number of central galaxies as a function of halo mass and redshift
    # Ngals = number of satellite galaxies as a function of halo mass and redshift
    # rhomatter =  current density in matter
    # log10mlow = log10 lower mass bound 
    # log10mhigh = log10 upper mass bound
    # spec = specifies the correlation function 
    # 'mm' = matter-matter
    # 'gasgas' = gas-gas
    # 'galgal' = galaxy-galaxy
    # 'mgas' = matter-gas
    # 'galm' = galaxy-matter
    # 'galgas' = galaxy-gas
    def _Ponehalo_binned(self,z,lnms,k,uk1,uk2,nfn,rhomatter,spec,log10mlow,log10mhigh,frequency=None):
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        #-------  
        
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        lnmsin=lnms[(lnms>=lnmlow)&(lnms<=lnmhigh)]
        uk1in=uk1[(lnms>=lnmlow)&(lnms<=lnmhigh),:,:]
        uk2in=uk2[(lnms>=lnmlow)&(lnms<=lnmhigh),:,:]
        nfnin=nfn[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalcin=Ngalc[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalsin=Ngals[(lnms>=lnmlow)&(lnms<=lnmhigh),:]

        onehalointegrand = np.zeros([lnmsin.shape[0],z.shape[0],k.shape[0]])

        if spec in ['mm','mgas','gasgas']:
            
            
            
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = uk1in[i,j,:]*uk2in[i,j,:]*nfnin[i,j]*np.exp(3.*lnmsin[i])/(rhomatter**2)
            

        '''
        if spec in ['mm','mgas','gasgas']:
            print("in function")#fiona
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = uk1in[i,j,:]*uk2in[i,j,:]*nfnin[i,j]*np.exp(3.*lnmsin[i])/(rhomatter**2)
        '''

        if spec in ['hgas']:
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = uk1in[i,j,:]*uk2in[i,j,:]*nfnin[i,j]*np.exp(2.*lnmsin[i])/(rhomatter*self.nbar_halo(z,log10mlow,log10mhigh))         

        if spec in ['galgal']:
            Ntot=Ngalcin+Ngalsin
            nbargaltot=self.Nbargal(lnmsin,z,nfnin,Ntot)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    if Ngalcin[i,j]>10**(-16):
                        onehalointegrand[i,j,:] = (nfnin[i,j]*np.exp(lnmsin[i])/(nbargaltot[j]**2))*(2.*Ngalsin[i,j]*uk1in[i,j,:] + (Ngalsin[i,j]**2)*(uk1in[i,j,:]**2.)/Ngalcin[i,j])
                    else:
                        onehalointegrand[i,j,:] = (nfnin[i,j]*np.exp(lnmsin[i])/(nbargaltot[j]**2))*(2.*Ngalsin[i,j]*uk1in[i,j,:])
                        
        if spec in ['galm','galgas']:
            Ntot=Ngalcin+Ngalsin
            nbargaltot=self.Nbargal(lnmsin,z,nfnin,Ntot)
            nbarc=self.Nbargal(lnmsin,z,nfnin,Ngalcin)
            nbars=self.Nbargal(lnmsin,z,nfnin,Ngalsin)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = (nfnin[i,j]*np.exp(2.*lnmsin[i])/(rhomatter*nbargaltot[j]))*(Ngalcin[i,j]*uk2in[i,j,:] + Ngalsin[i,j]*uk1in[i,j,:]*uk2in[i,j,:])
      
       
           
        

        Ponehalo = np.zeros([z.shape[0],k.shape[0]])
       
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                if k[q]>.01:
                    Ponehalo[j,q] = np.trapz(onehalointegrand[:,j,q],lnmsin)
                else:
                    Ponehalo[j,q] = 10.**(-16.)
        

        if self.onehdamping:
            Ponehalo *= (1.-np.exp(-(k/self.onehdamping_kstar)**2.))

        return Ponehalo



    # Function to calculate the two-halo auto and cross power spectrum for matter, gas, galaxies.
    # z =  redshifts
    # lnms = natural log of halo masses
    # k = comoving wavenumbers
    # uk1 = FT of galaxy run (if galaxy part of correlation), otherwise mass or gas
    # uk2 = FT of mass or gas.
    # nfn = halo mass function
    # Ngalc = number of central galaxies as a function of halo mass and redshift
    # Ngals = number of satellite galaxies as a function of halo mass and redshift
    # rhomatter =  current density in matter
    # halobias = precomputed linear halo bias
    # log10mlow = log10 lower mass bound 
    # log10mhigh = log10 upper mass bound
    # spec = specifies the correlation function 
    # 'mm' = matter-matter
    # 'gasgas' = gas-gas
    # 'galgal' = galaxy-galaxy
    # 'mgas' = matter-gas
    # 'galm' = galaxy-matter
    # 'galgas' = galaxy-gas
    # log10mlow2 = log10 lower mass bound for cross mass bin spectra
    # log10mhigh2 = log10 upper mass bound for cross mass bin spectra
    def _Ptwohalo_binned(self,z,lnms,k,pk,uk1,uk2,nfn,rhomatter,halobias,spec,log10mlow,log10mhigh,log10mlow2=None,log10mhigh2=None):
        #------- Number of10000 satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        print("in two halo binned")#Fiona
        print(spec)
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        #-------
        if (log10mlow2==None)or(log10mhigh2==None):
            log10mlow2 = log10mlow
            log10mhigh2 = log10mhigh
        
        consistency=np.zeros([z.shape[0]])

        lnmlow1=np.log(10.**log10mlow)
        lnmhigh1=np.log(10.**log10mhigh)

        lnmsin1=lnms[(lnms>=lnmlow1)&(lnms<=lnmhigh1)]
        nfnin1=nfn[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:]
        Ngalcin1=Ngalc[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:]
        Ngalsin1=Ngals[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:]
        halobiasin1=halobias[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:]
        uk1in=uk1[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:,:]

    
        lnmlow2=np.log(10.**log10mlow2)
        lnmhigh2=np.log(10.**log10mhigh2)
        lnmsin2=lnms[(lnms>=lnmlow2)&(lnms<=lnmhigh2)]
        nfnin2=nfn[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:]
        Ngalcin2=Ngalc[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:]
        Ngalsin2=Ngals[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:]
        halobiasin2=halobias[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:]
        uk2in=uk2[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:,:]

        twohalointegrand1 = np.zeros([lnmsin1.shape[0],z.shape[0],k.shape[0]])
        twohalointegrand2 = np.zeros([lnmsin2.shape[0],z.shape[0],k.shape[0]])
        
        

        if spec in ['mm','mgas','gasgas']:  
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin1.shape[0]):
                    twohalointegrand1[i,j,:] = uk1in[i,j,:]*halobiasin1[i,j]*nfnin1[i,j]*np.exp(2.*lnmsin1[i])/(rhomatter)
                for i in range(0,lnmsin2.shape[0]):
                    twohalointegrand2[i,j,:] = uk2in[i,j,:]*halobiasin2[i,j]*nfnin2[i,j]*np.exp(2.*lnmsin2[i])/(rhomatter)
                    

        
        if spec in ['galgal']:
            Ntot1=Ngalcin1+Ngalsin1
            nbargaltot1=self.Nbargal(lnmsin1,z,nfnin1,Ntot1)
            Ntot2=Ngalcin2+Ngalsin2
            nbargaltot2=self.Nbargal(lnmsin2,z,nfnin2,Ntot2)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin1.shape[0]):
                    twohalointegrand1[i,j,:] = halobiasin1[i,j]*nfnin1[i,j]*np.exp(lnmsin1[i])*(Ngalcin1[i,j]+Ngalsin1[i,j]*uk1in[i,j,:])/nbargaltot1[j]
                for i in range(0,lnmsin2.shape[0]):
                    twohalointegrand2[i,j,:] = halobiasin2[i,j]*nfnin2[i,j]*np.exp(lnmsin2[i])*(Ngalcin2[i,j]+Ngalsin2[i,j]*uk2in[i,j,:])/nbargaltot2[j]

        if spec in ['galm','galgas']:
            Ntot1=Ngalcin1+Ngalsin1
            nbargaltot1=self.Nbargal(lnmsin1,z,nfnin1,Ntot1)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin1.shape[0]):
                    twohalointegrand1[i,j,:] = halobiasin1[i,j]*nfnin1[i,j]*np.exp(lnmsin1[i])*(Ngalcin1[i,j]+Ngalsin1[i,j]*uk1in[i,j,:])/nbargaltot1[j]
                for i in range(0,lnmsin2.shape[0]):
                    twohalointegrand2[i,j,:] = uk2in[i,j,:]*halobiasin2[i,j]*nfnin2[i,j]*np.exp(2.*lnmsin2[i])/(rhomatter)

        Ptwohalo = np.zeros([z.shape[0],k.shape[0]])
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                Ptwohalo[j,q] = pk[j,q]*np.trapz(twohalointegrand1[:,j,q],lnmsin1)*np.trapz(twohalointegrand2[:,j,q],lnmsin2)

        return Ptwohalo



    # Function to calculate the one-halo auto and cross power spectrum for matter, gas, galaxies.
    # z =  redshifts
    # lnms = natural log of halo masses
    # k = comoving wavenumbers
    # uk1 = FT of galaxy run (if galaxy part of correlation), otherwise mass or gas
    # uk2 = FT of mass or gas.
    # nfn = halo mass function
    # Ngalc = number of central galaxies as a function of halo mass and redshift
    # Ngals = number of satellite galaxies as a function of halo mass and redshift
    # rhomatter =  current density in matter
    # spec = specifies the correlation function 
    # 'mm' = matter-matter
    # 'gasgas' = gas-gas
    # 'galgal' = galaxy-galaxy
    # 'mgas' = matter-gas
    # 'galm' = galaxy-matter
    # 'galgas' = galaxy-gas
    
    # 'CIBCIB'=CIB-CIB
    # 'CIBgas'=CIB-gas
    

    
    def _Ponehalo(self,z,lnms,k,uk1,uk2,nfn,rhomatter,spec,frequency=None):#Fiona: added optional frequency argument for CIB
        
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        #-------  
        
        onehalointegrand = np.zeros([self.npts,z.shape[0],k.shape[0]])

        if spec in ['mm','mgas','gasgas']:
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    onehalointegrand[i,j,:] = uk1[i,j,:]*uk2[i,j,:]*nfn[i,j]*np.exp(3.*lnms[i])/(rhomatter**2)

        if spec in ['galgal']:
            Ntot=Ngalc+Ngals
            nbargaltot=self.Nbargal(lnms,z,nfn,Ntot)

            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):                
                    if Ngalc[i,j]>10**(-16):
                        onehalointegrand[i,j,:] = (nfn[i,j]*np.exp(lnms[i])/(nbargaltot[j]**2))*(2.*Ngals[i,j]*uk1[i,j,:] + (Ngals[i,j]**2)*(uk1[i,j,:]**2.)/Ngalc[i,j])
                    else:
                        onehalointegrand[i,j,:] = (nfn[i,j]*np.exp(lnms[i])/(nbargaltot[j]**2))*(2.*Ngals[i,j]*uk1[i,j,:])

        if spec in ['galm','galgas']:
            Ntot=Ngalc+Ngals
            nbargaltot=self.Nbargal(lnms,z,nfn,Ntot)
            nbarc=self.Nbargal(lnms,z,nfn,Ngalc)
            nbars=self.Nbargal(lnms,z,nfn,Ngals)

            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    onehalointegrand[i,j,:] = (nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*nbargaltot[j]))*(Ngalc[i,j]*uk2[i,j,:] + Ngals[i,j]*uk1[i,j,:]*uk2[i,j,:])

        if spec in ['tSZtSZ']:
            sigmaT= 6.6524587158e-29
            mElect= 4.579881126194068e-61
            c=299792458.0
            tsz_factor=4*np.pi*(sigmaT/(mElect*c**2))*(1+self.z)**2*kszpsz_config.T_CMB

            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    onehalointegrand[i,j,:] = tsz_factor[j]**2*nfn[i,j]*uk1[i,j,:]*uk2[i,j,:]*np.exp(lnms[i])

         
        if spec in ['tSZgas']:
            
            sigmaT= 6.6524587158e-29
            mElect= 4.579881126194068e-61
            c=299792458.0
            tsz_factor=4*np.pi*(sigmaT/(mElect*c**2))*1/(1+self.z)**2*kszpsz_config.T_CMB

            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    onehalointegrand[i,j,:] = tsz_factor[j]**1*nfn[i,j]*uk1[i,j,:]*uk2[i,j,:]*1/rhomatter*np.exp(2.*lnms[i])



        if spec in ['CIBCIB']: #Fiona 
        #need uk1 to correspond to ukm; do not need uk2.
            t1=time.time()

            #satflux=self.satflux_interpolated(frequency)# define a function to interpolate the sat_flux.
            if kszpsz_config.CIB_model=="minimally_empiricial":
                file=np.load("stellarmasses_new.npz")
                mstellars=file["mstellars"]
                precompzs=file["zs"]
                precompms=file["mhalos"]
            
            
                mstellar_interp=interp2d(precompms,precompzs,mstellars)
                mstellars=mstellar_interp(np.exp(lnms),z)
            
                central_flux=CIB.Scentral(frequency,z[:,np.newaxis],mstellars)
                central_flux=np.transpose(central_flux)
                
                satflux=np.transpose(CIB.sat_flux(frequency,np.exp(lnms),z))
           
            elif kszpsz_config.CIB_model=="Planck":
                central_flux=(CIB_Planck.Scentral(frequency,z,np.exp(lnms)[:,np.newaxis]))
                satflux=((CIB_Planck.satellite_intensity(frequency,z,np.exp(lnms))))
                
            elif kszpsz_config.CIB_model=="Websky":
                central_flux=np.transpose(CIB_Websky.Scentral(frequency,z[:,np.newaxis],np.exp(lnms)))
                satflux=((CIB_Websky.satellite_intensity(frequency,z,np.exp(lnms))))
                #np.save('./test_satfluxbef.npy',satflux)
                #np.save('./test_centralfluxbef.npy',central_flux)
                satflux[central_flux==0]=0
                central_flux[satflux>CIB_Websky.Scut(frequency)]=0

                satflux[satflux>CIB_Websky.Scut(frequency)]=0
            #np.save('./test_satflux.npy',satflux)
            #np.save('./test_centralflux.npy',central_flux)

            print(central_flux.shape,satflux.shape,'saved')
            onehalointegrand=remote_spectra.chifromz(z[np.newaxis,:,np.newaxis])**4*(np.exp(lnms[:,np.newaxis,np.newaxis])*(2*nfn[:,:,np.newaxis]*central_flux[:,:,np.newaxis]*satflux[:,:,np.newaxis]*uk1[:,:,:]+nfn[:,:,np.newaxis]*satflux[:,:,np.newaxis]**2*uk1[:,:,:]**2 ) ) #check this
            
           # print("done in",time.time()-t1,"seconds")
        if spec in ['CIBgas']: #Fiona 
            if kszpsz_config.CIB_model=="minimally_empiricial":
                satflux=np.transpose(CIB.sat_flux(frequency,np.exp(self.lnms),z)  )          
                central_flux=np.zeros((len(z),len(lnms)))
                file=np.load("stellarmasses_new.npz")
                mstellars=file["mstellars"]
                precompzs=file["zs"]
                precompms=file["mhalos"]
                mstellar_interp=interp2d(precompms,precompzs,mstellars)
                mstellars=mstellar_interp(np.exp(lnms),z)
            
                central_flux=CIB.Scentral(frequency,z[:,np.newaxis],mstellars)
                central_flux=np.transpose(central_flux)

            elif kszpsz_config.CIB_model=="Planck":
                central_flux=(CIB_Planck.Scentral(frequency,z,np.exp(lnms)[:,np.newaxis]))
                satflux=((CIB_Planck.satellite_intensity(frequency,z,np.exp(lnms))))
               

            elif kszpsz_config.CIB_model=="Websky":
                central_flux=np.transpose(CIB_Websky.Scentral(frequency,z[:,np.newaxis],np.exp(lnms)))
                satflux=((CIB_Websky.satellite_intensity(frequency,z,np.exp(lnms))))
                satflux[central_flux==0]=0
                central_flux[satflux>CIB_Websky.Scut(frequency)]=0

                satflux[satflux>CIB_Websky.Scut(frequency)]=0
                
                
                
            else:
                print ("problem",kszpsz_config.CIB_model)
  
            
           
          #  onehalointegrand =remote_spectra.chifromz(z[np.newaxis,:,np.newaxis])**2*
          #  np.exp(lnms[:,np.newaxis,np.newaxis])*(nfn[:,:,np.newaxis]*np.exp(lnms[:,np.newaxis,np.newaxis])/(rhomatter))*(nfn[:,:,np.newaxis]*central_flux[:,:,np.newaxis] + satflux[:,:,np.newaxis]*uk1[:,:,:])*uk2[:,:,:]
            onehalointegrand =remote_spectra.chifromz(z[np.newaxis,:,np.newaxis])**2*np.exp(lnms[:,np.newaxis,np.newaxis])*(nfn[:,:,np.newaxis]*np.exp(lnms[:,np.newaxis,np.newaxis])/(rhomatter))*(central_flux[:,:,np.newaxis] + satflux[:,:,np.newaxis]*uk1[:,:,:])*uk2[:,:,:]


        Ponehalo = np.zeros([z.shape[0],k.shape[0]])
        t1=time.time()
        print("integration starting")
        Ponehalo[:,k>0.01]=np.trapz(onehalointegrand[:,:,k>0.01],lnms,axis=0)
        Ponehalo[:,k<0.01]= Ponehalo[:,k<0.01]*10.**(-16.)
        print("integration done in",time.time()-t1,"seconds")

        
        '''
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                if k[q]>.01:
                    Ponehalo[j,q] = np.trapz(onehalointegrand[:,j,q],lnms)
                else:
                    Ponehalo[j,q] = 10.**(-16.)
           
        '''
        if self.onehdamping:
            Ponehalo *= (1.-np.exp(-(k/self.onehdamping_kstar)**2.))                    
        return Ponehalo



    # Function to calculate the two-halo auto and cross power spectrum for matter, gas, galaxies.
    # z =  redshifts
    # lnms = natural log of halo masses
    # k = comoving wavenumbers
    # uk1 = FT of galaxy run (if galaxy part of correlation), otherwise mass or gas
    # uk2 = FT of mass or gas.
    # nfn = halo mass function
    # Ngalc = number of central galaxies as a function of halo mass and redshift
    # Ngals = number of satellite galaxies as a function of halo mass and redshift
    # rhomatter =  current density in matter
    # halobias = precomputed linear halo bias
    # spec = specifies the correlation function 
    # 'mm' = matter-matter
    # 'gasgas' = gas-gas
    # 'galgal' = galaxy-galaxy
    # 'mgas' = matter-gas
    # 'galm' = galaxy-matter
    # 'galgas' = galaxy-gas
    def _Ptwohalo(self,z,lnms,k,pk,uk1,uk2,nfn,rhomatter,halobias,spec,frequency=None):#Fiona: added optional frequency argument
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        #-------       
        twohalointegrand1 = np.zeros([self.npts,z.shape[0],k.shape[0]])
        twohalointegrand2 = np.zeros([self.npts,z.shape[0],k.shape[0]])
        consistency=np.zeros([z.shape[0]])

        for j in range(0,z.shape[0]):
            consistency[j] = np.trapz(np.exp(2.*lnms)*nfn[:,j]*halobias[:,j]/(self.omegam*self.rhocrit),lnms)

        if spec in ['mm','mgas','gasgas']:    
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    twohalointegrand1[i,j,:] = uk1[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j])
                    twohalointegrand2[i,j,:] = uk2[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j]) 

        if spec in ['galgal']:
            Ntot=Ngalc+Ngals
            nbargaltot=self.Nbargal(lnms,z,nfn,Ntot)
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    twohalointegrand1[i,j,:] = halobias[i,j]*nfn[i,j]*np.exp(lnms[i])*(Ngalc[i,j]+Ngals[i,j]*uk1[i,j,:])/nbargaltot[j]
                    twohalointegrand2[i,j,:] = halobias[i,j]*nfn[i,j]*np.exp(lnms[i])*(Ngalc[i,j]+Ngals[i,j]*uk2[i,j,:])/nbargaltot[j]

        if spec in ['galm','galgas']:
            Ntot=Ngalc+Ngals
            nbargaltot=self.Nbargal(lnms,z,nfn,Ntot)
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    twohalointegrand1[i,j,:] = halobias[i,j]*nfn[i,j]*np.exp(lnms[i])*(Ngalc[i,j]+Ngals[i,j]*uk1[i,j,:])/nbargaltot[j]
                    twohalointegrand2[i,j,:] = uk2[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j])

        if spec in ['CIBCIB']: #Fiona
            
            if kszpsz_config.CIB_model=="minimally_empiricial":

                central_flux=np.zeros((len(z),len(lnms)))
                file=np.load("stellarmasses.npz")
                mstellars=file["mstellars"]
                precompzs=file["zs"]
                precompms=file["mhalos"]
                mstellar_interp=interp2d(precompms,precompzs,mstellars)
                mstellars=mstellar_interp(np.exp(lnms),z)
            
                central_flux=CIB.Scentral(frequency,z[:,np.newaxis],mstellars)
                central_flux=np.transpose(central_flux)
            
                satflux=np.transpose(CIB.sat_flux(frequency,np.exp(self.lnms),z))
            elif kszpsz_config.CIB_model=="Planck":
                central_flux=(CIB_Planck.Scentral(frequency,z,np.exp(lnms)[:,np.newaxis]))
                satflux=((CIB_Planck.satellite_intensity(frequency,z,np.exp(lnms))))
            elif kszpsz_config.CIB_model=="Websky":
                central_flux=np.transpose(CIB_Websky.Scentral(frequency,z[:,np.newaxis],np.exp(lnms)))
              #  print("cf",central_flux)
                satflux=((CIB_Websky.satellite_intensity(frequency,z,np.exp(lnms))))
                
                satflux[central_flux==0]=0
                central_flux[satflux>CIB_Websky.Scut(frequency)]=0

                satflux[satflux>CIB_Websky.Scut(frequency)]=0
                
                
            twohalointegrand1 = remote_spectra.chifromz(z[np.newaxis,:,np.newaxis])**2*np.exp(lnms[:,np.newaxis,np.newaxis])*(self.nfn[:,:,np.newaxis]*self.halobias[:,:,np.newaxis]*(central_flux[:,:,np.newaxis]+satflux[:,:,np.newaxis]) )
            twohalointegrand1=twohalointegrand1*np.ones(k.shape)
            twohalointegrand2 = twohalointegrand1
    
        if spec in ['CIBgas']: #Fiona
            if kszpsz_config.CIB_model=="minimally_empiricial":
                satflux=np.transpose(CIB.sat_flux(frequency,np.exp(self.lnms),z))
                central_flux=np.zeros((len(z),len(lnms)))
                file=np.load("stellarmasses.npz")
                mstellars=file["mstellars"]
                precompzs=file["zs"]
                precompms=file["mhalos"]
                mstellar_interp=interp2d(precompms,precompzs,mstellars)
                mstellars=mstellar_interp(np.exp(lnms),z)
                central_flux=CIB.Scentral(frequency,z[:,np.newaxis],mstellars)
                central_flux=np.transpose(central_flux)
            elif kszpsz_config.CIB_model=="Planck":
                central_flux=(CIB_Planck.Scentral(frequency,z,np.exp(lnms)[:,np.newaxis]))
                satflux=((CIB_Planck.satellite_intensity(frequency,z,np.exp(lnms))))
            elif kszpsz_config.CIB_model=="Websky":
                central_flux=np.transpose(CIB_Websky.Scentral(frequency,z[:,np.newaxis],np.exp(lnms)))
                satflux=((CIB_Websky.satellite_intensity(frequency,z,np.exp(lnms))))
                satflux[central_flux==0]=0
                central_flux[satflux>CIB_Websky.Scut(frequency)]=0

                satflux[satflux>CIB_Websky.Scut(frequency)]=0
                
            twohalointegrand1 = remote_spectra.chifromz(z[np.newaxis,:,np.newaxis])**2*np.exp(lnms[:,np.newaxis,np.newaxis])*(self.nfn[:,:,np.newaxis]*self.halobias[:,:,np.newaxis]*(central_flux[:,:,np.newaxis]+satflux[:,:,np.newaxis]) )
            twohalointegrand1= twohalointegrand1*np.ones(k.shape)
            twohalointegrand2 = uk2[:,:,:]*halobias[:,:,np.newaxis]*nfn[:,:,np.newaxis]*np.exp(2.*lnms[:,np.newaxis,np.newaxis])/(rhomatter*consistency[np.newaxis,:,np.newaxis]) #note why was it originally 2*lnms?? cause it is MdM?

        
        Ptwohalo = np.zeros([z.shape[0],k.shape[0]])
        t1=time.time()
        
        Ptwohalo= pk*np.trapz(twohalointegrand1,lnms,axis=0)*np.trapz(twohalointegrand2,lnms,axis=0)
        
        
        '''
        for j in range(0,z.shape[0]):
            #if j==0:
            
            for q in range(0,k.shape[0]):
               
                Ptwohalo[j,q] = pk[j,q]*np.trapz(twohalointegrand1[:,j,q],lnms)*np.trapz(twohalointegrand2[:,j,q],lnms)
        '''
        
        print("integrated two halo in",time.time()-t1,"seconds")
            
           
        return Ptwohalo


    
    #version of P_ge where we want all electrons (no mass cut), but only the galaxies within the mass bin
    def _Ptwohalo_halobinned_gasunbinned(self,z,lnms,k,pk,uk1,uk2,nfn,rhomatter,halobias,spec,log10mlow,log10mhigh):
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral_binned(self.lnms,self.z[j],self.mthreshHOD[j],log10mlow,log10mhigh)
            Ngals[:,j]=self.Nsatellite_binned(self.lnms,self.z[j],self.mthreshHOD[j],log10mlow,log10mhigh)
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        #-------    
        twohalointegrand1 = np.zeros([self.npts,z.shape[0],k.shape[0]])
        twohalointegrand2 = np.zeros([self.npts,z.shape[0],k.shape[0]])
        consistency=np.zeros([z.shape[0]])

        for j in range(0,z.shape[0]):
            consistency[j] = np.trapz(np.exp(2.*lnms)*nfn[:,j]*halobias[:,j]/(self.omegam*self.rhocrit),lnms)

        if spec in ['mgas']:    
            lnmlow=np.log(10.**log10mlow)
            lnmhigh=np.log(10.**log10mhigh)
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    if (lnms[i]>=lnmlow) and (lnms[i]<lnmhigh):
                        twohalointegrand1[i,j,:] = uk1[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j])
                    twohalointegrand2[i,j,:] = uk2[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j]) #uk2 is the gas

        if spec in ['hgas']: 
            lnmlow=np.log(10.**log10mlow)
            lnmhigh=np.log(10.**log10mhigh)
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    if (lnms[i]>=lnmlow) and (lnms[i]<lnmhigh):
                        twohalointegrand1[i,j,:] = uk1[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(lnms[i])/self.nbar_halo(z,log10mlow,log10mhigh)  #this equals the binned bias
                    twohalointegrand2[i,j,:] = uk2[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j]) #uk2 is the gas            
                    
        if spec in ['galgas']:
            Ntot=Ngalc+Ngals
            nbargaltot=self.Nbargal(lnms,z,nfn,Ntot)
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    twohalointegrand1[i,j,:] = halobias[i,j]*nfn[i,j]*np.exp(lnms[i])*(Ngalc[i,j]+Ngals[i,j]*uk1[i,j,:])/nbargaltot[j]
                    twohalointegrand2[i,j,:] = uk2[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j])

        Ptwohalo = np.zeros([z.shape[0],k.shape[0]])
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                Ptwohalo[j,q] = pk[j,q]*np.trapz(twohalointegrand1[:,j,q],lnms)*np.trapz(twohalointegrand2[:,j,q],lnms)

        return Ptwohalo


    


    ################################## convinient wrappers of the previous functions


    #this is just the shot noise. 
    def P_hh_1h(self,ks,zs,logmlow=None,logmhigh=None):
        Phh1h = np.ones( (self.pk.shape) ) * 1./self.nbar_halo(zs,logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Phh1h),bounds_error=False,fill_value=0)
        Phh1h = np.power(10.0, Pinterp(np.log10(ks)))
        return Phh1h

    #set second mass value for cross spectra
    def P_hh_2h(self,ks,zs,logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None):
        if logmlow2 == None:
            logmlow2 = logmlow
            logmhigh2 = logmhigh
        Phh2h = self.pk * self.bias_halo(zs,logmlow,logmhigh) * self.bias_halo(zs,logmlow2,logmhigh2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Phh2h),bounds_error=False,fill_value=0)
        Phh2h = np.power(10.0, Pinterp(np.log10(ks)))
        return Phh2h    

    def P_mm_1h(self,ks,zs,logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,self.ukm,self.nfn,self.rhomatter,'mm')
        else:
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,self.ukm,self.ukm,self.nfn,self.rhomatter,'mm',logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo
    
    #set second mass value for cross spectra
    def P_mm_2h(self,ks,zs,logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'mm')
        else:
            Ptwohalo = self._Ptwohalo_binned(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'mm',logmlow,logmhigh,logmlow2,logmhigh2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))            
        return Ptwohalo   
    
    def P_mattermatter_2h(self,ks,zs,logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'mm')
        else:
            Ptwohalo = self._Ptwohalo_binned(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'mm',logmlow,logmhigh,logmlow2,logmhigh2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))            
        return Ptwohalo   

    def P_gg_1h(self,ks,zs,logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,self.ukm,self.nfn,self.rhomatter,'galgal')
        else:
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,self.ukm,self.ukm,self.nfn,self.rhomatter,'galgal',logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo

    def P_gg_2h(self,ks,zs,logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'galgal')
        else:
            Ptwohalo = self._Ptwohalo_binned(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'galgal',logmlow,logmhigh,logmlow2,logmhigh2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))  
        return Ptwohalo

    def P_gm_1h(self,ks,zs,logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,self.ukm,self.nfn,self.rhomatter,'galm')
        else:
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,self.ukm,self.ukm,self.nfn,self.rhomatter,'galm',logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo

    def P_gm_2h(self,ks,zs,logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'galm')
        else:
            Ptwohalo = self._Ptwohalo_binned(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'galm',logmlow,logmhigh,logmlow2,logmhigh2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))  
        return Ptwohalo

    def P_CIBCIB_1h(self,ks,zs,frequency,logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):#Fiona
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        t1=time.time()
        print("finding one halo")
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        print("set up grid functions in ",time.time()-t1,"seconds")
        t1=time.time()
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,self.ukm,self.nfn,self.rhomatter,'CIBCIB',frequency)
        else:
            #figure out this. so far i just have copy and pasted from above. what difference will this make?
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,self.ukm,self.nfn,self.rhomatter,'CIBCIB',frequency)

        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        
        
       # print("done in",time.time()-t1,"seconds")
        return Ponehalo
            
    def P_CIBCIB_2h(self,ks,zs,frequency,logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):#Fiona
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        if (logmlow==None) or (logmhigh==None):
            
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'CIBCIB',frequency)
        else:
            #figure out this. so far i just have copy and pasted from above. what difference will this make?
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,self.ukm,self.nfn,self.rhomatter,self.halobias,'CIBCIB',frequency)

        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ptwohalo

    def P_tSZtSZ_1h(self,ks,zs,logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001

        self._setup_k_z_m_grid_functions(zs,need_ykm=True)
      

        
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ykm,self.ykm,self.nfn,self.rhomatter,'tSZtSZ')
        else:
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ykm,self.ykm,self.nfn,self.rhomatter,'tSZtSZ')

           # print("problem")
       #     Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,ukgas,ukgas,self.nfn,self.rhomatter,'gasgas',logmlow,logmhigh)
        
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo
    
    def P_tSZe_1h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001

        self._setup_k_z_m_grid_functions(zs,need_ukm=True,need_ykm=True)
        ukgas = self._get_gas_profile(gasprofile)

        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ykm,ukgas,self.nfn,self.rhomatter,'tSZgas')
        else:
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ykm,ukgas,self.nfn,self.rhomatter,'tSZgas')

            print("problem")
       #     Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,ukgas,ukgas,self.nfn,self.rhomatter,'gasgas',logmlow,logmhigh)
        
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo
        

    #gasprofile: universal, AGN, SH
    def P_ee_1h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001

        self._setup_k_z_m_grid_functions(zs)
        ukgas = self._get_gas_profile(gasprofile)
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,ukgas,ukgas,self.nfn,self.rhomatter,'gasgas')
        else:
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,ukgas,ukgas,self.nfn,self.rhomatter,'gasgas',logmlow,logmhigh)
        
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo

    def P_ee_2h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        ukgas = self._get_gas_profile(gasprofile)
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,ukgas,ukgas,self.nfn,self.rhomatter,self.halobias,'gasgas')
        else:
            Ptwohalo = self._Ptwohalo_binned(self.z,self.lnms,self.k,self.pk,ukgas,ukgas,self.nfn,self.rhomatter,self.halobias,'gasgas',logmlow,logmhigh,logmlow2,logmhigh2)

        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks))) 

        return Ptwohalo

    def P_he_1h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        ukgas = self._get_gas_profile(gasprofile)
        ukhalo = np.ones( (ukgas.shape) )
        if (logmlow!=None) or (logmhigh!=None):
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,ukhalo,ukgas,self.nfn,self.rhomatter,'hgas',logmlow,logmhigh) #all e is the same as the ordinary binned version.
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo
    
    def P_he_2h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        ukgas = self._get_gas_profile(gasprofile)
        ukhalo = np.ones( (ukgas.shape) )
        if (logmlow!=None) or (logmhigh!=None):
            Ptwohalo = self._Ptwohalo_halobinned_gasunbinned(self.z,self.lnms,self.k,self.pk,ukhalo,ukgas,self.nfn,self.rhomatter,self.halobias,'hgas',logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))  
        return Ptwohalo   

    def P_me_1h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        ukgas = self._get_gas_profile(gasprofile)
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,ukgas,self.nfn,self.rhomatter,'mgas')
        else:
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,self.ukm,ukgas,self.nfn,self.rhomatter,'mgas',logmlow,logmhigh) 
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo
    
    def P_me_2h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        ukgas = self._get_gas_profile(gasprofile)
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,ukgas,self.nfn,self.rhomatter,self.halobias,'mgas')
        else:
            Ptwohalo = self._Ptwohalo_halobinned_gasunbinned(self.z,self.lnms,self.k,self.pk,self.ukm,ukgas,self.nfn,self.rhomatter,self.halobias,'mgas',logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))  
        return Ptwohalo
    
    #in P_ge we want all electrons (no mass cut), but only the galaxies within the mass bin
    def P_ge_1h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        ukgas = self._get_gas_profile(gasprofile)
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,ukgas,self.nfn,self.rhomatter,'galgas')
        else:
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,self.ukm,ukgas,self.nfn,self.rhomatter,'galgas',logmlow,logmhigh) 
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo
    
    #in P_ge we want all electrons (no mass cut), but only the galaxies within the mass bin
    def P_ge_2h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        ukgas = self._get_gas_profile(gasprofile)
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,ukgas,self.nfn,self.rhomatter,self.halobias,'galgas')
        else:
            Ptwohalo = self._Ptwohalo_halobinned_gasunbinned(self.z,self.lnms,self.k,self.pk,self.ukm,ukgas,self.nfn,self.rhomatter,self.halobias,'galgas',logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))  
        return Ptwohalo
    
    
    def P_CIBe_1h(self,ks,zs,frequency,gasprofile='universal',logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):#Fiona
        zs = np.atleast_1d(zs)
        print("getting one halo cross")
        t1=time.time()
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        ukgas = self._get_gas_profile(gasprofile)
      #  frequency=545
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,ukgas,self.nfn,self.rhomatter,'CIBgas',frequency)
        else:
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukm,ukgas,self.nfn,self.rhomatter,'CIBgas',frequency) 
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        print("found in",time.time()-t1,"seconds")
        return Ponehalo
    
    #in P_ge we want all electrons (no mass cut), but only the galaxies within the mass bin
    def P_CIBe_2h(self,ks,zs,frequency,gasprofile='universal',logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):#Fiona
        zs = np.atleast_1d(zs)
        print("getting two halo cross")
        t1=time.time()
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        ukgas = self._get_gas_profile(gasprofile)
      #  frequency=353
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,ukgas,self.nfn,self.rhomatter,self.halobias,'CIBgas',frequency)
        else:
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukm,ukgas,self.nfn,self.rhomatter,self.halobias,'CIBgas',frequency)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))  
        print("found in",time.time()-t1,"seconds")

        return Ptwohalo
    
    
        

    def bias_halo(self,zs,logmlow,logmhigh):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)            
        lnmlow=np.log(10.**logmlow)
        lnmhigh=np.log(10.**logmhigh)
        lnmsin=self.lnms[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh)]
        nfnin=self.nfn[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh),:]
        halobiasin=self.halobias[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh),:]
        bhalo = np.zeros([self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            bhalo[j]=np.trapz((np.exp(lnmsin))*nfnin[:,j]*halobiasin[:,j],lnmsin)/np.trapz((np.exp(lnmsin))*nfnin[:,j],lnmsin) #as eq 49 
        return bhalo
    
    def bias_galaxy(self,zs,logmlow,logmhigh,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        #-------           
        return self._galaxybias(self.lnms,zs,self.nfn,Ngalc,Ngals,self.halobias,logmlow,logmhigh)

    def nbar_halo(self,zs,logmlow,logmhigh):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)        
        lnmlow=np.log(10.**logmlow)
        lnmhigh=np.log(10.**logmhigh)
        lnmsin=self.lnms[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh)]
        nfnin=self.nfn[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh),:]
        nhalobinned=np.zeros(self.z.shape[0])
        for j in range(0,self.z.shape[0]):
            nhalobinned[j]=np.trapz(np.exp(lnmsin)*nfnin[:,j],lnmsin)
        return nhalobinned
    
    def nbar_galaxy(self,zs,logmlow,logmhigh,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        #-------          
        return self.Nbargal_binned(self.lnms,zs,self.nfn,Ngalc+Ngals,logmlow,logmhigh)
    
    def CIB_shot_noise(self,nu):
        if kszpsz_config.CIB_model=="minimally_empiricial":
            file=np.load("stellarmasses_new.npz")
            mstellars=file["mstellars"]
            precompzs=file["zs"]
            precompms=file["mhalos"]
            mstellar_interp=interp2d(precompms,precompzs,mstellars)
            mstellars=mstellar_interp(np.exp(self.lnms),self.z)
        
            sn=CIB.shot_noise(self.nfn,nu,self.z,np.exp(self.lnms),np.transpose(mstellars))
        elif kszpsz_config.CIB_model=="Planck" or kszpsz_config.CIB_model=="Websky":
            sn=CIB_Planck.sn(nu,nu)

            
        return sn
    

        
        

def main(argv):    
    
    #------- Calculate halo model
    halomodel = HaloModel()

    #sampling in z and k
    zmin=10**(-3)      
    zmax=2 #6              
    nz=20 #50           
    #zs = np.linspace(zmin,zmax,nz)
    zs = np.array( [0.85] ) # [0.01,0.3,0.57,1.5]
    kmin = 0.1 #1e-3 #0.1 #1e-5
    kmax = 10 #1e-1 #10.
    nk = 2000 #100
    ks = np.logspace(np.log10(kmin),np.log10(kmax),num=nk)
    z_id=0 #index in z_id shift array

    # #------------- get number densities and biases

    massbins =  np.arange(10.,16.05,0.25)
    ngalMpc3 = np.zeros(massbins.shape[0])
    ngalMpc3_halo = np.zeros(massbins.shape[0])
    galaxybias = np.zeros(massbins.shape[0])
    halobias = np.zeros(massbins.shape[0])
    for m_id,m in enumerate(massbins[:-1]):
        log10mlow = massbins[m_id]
        log10mhigh = massbins[m_id+1]
        ngalMpc3[m_id] = halomodel.nbar_galaxy(zs,log10mlow,log10mhigh)[0]
        galaxybias[m_id] = halomodel.bias_galaxy(zs,log10mlow,log10mhigh)[0]
        ngalMpc3_halo[m_id] = halomodel.nbar_halo(zs,log10mlow,log10mhigh)[0]
        halobias[m_id] = halomodel.bias_halo(zs,log10mlow,log10mhigh)[0]

    fig=plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(111)
    ax1.plot(massbins,ngalMpc3,ls='solid',label=r'nbar_gal')
    ax1.plot(massbins,ngalMpc3_halo,ls='solid',label=r'nbar_halo') 
    ax1.set_yscale('log')
    plt.ylim([1e-13,1e-1])
    plt.legend(loc=1,frameon=False)
    plt.grid()
    fig.tight_layout()
    plt.show()
    fig.savefig('plots/ngal.pdf')

    fig=plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(111)
    ax1.plot(massbins,galaxybias,ls='solid',label=r'b_gal')
    ax1.plot(massbins,halobias,ls='dashed',label=r'b_halo') 
    ax1.set_yscale('log')
    plt.legend(loc=1,frameon=False)
    plt.grid()
    fig.tight_layout()
    plt.show()
    fig.savefig('plots/galaxybias.pdf')
     
    #return

            
    #-------------test bias vs mass

    # #mass binning:
    # logmasses = np.array([12.,13.,14.,15.,16.])
    # z_id = 0
    # for lm_id, lm in enumerate(logmasses[:-1]):
    #     log10mlow = logmasses[lm_id]
    #     log10mhigh = logmasses[lm_id+1]
    #     pmm1h = halomodel.P_mm_1h(ks,zs)[z_id,0]
    #     pmm2h = halomodel.P_mm_2h(ks,zs)[z_id,0]
    #     pgg1h = halomodel.P_gg_1h(ks,zs,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,0]
    #     pgg2h = halomodel.P_gg_2h(ks,zs,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,0]
    #     pge2h2 = halomodel.P_ge_2h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,0]
    #     #bias_ps = np.sqrt((pmm1h+pmm2h)/(pgg1h+pgg2h))
    #     bias_ps = np.sqrt(pgg2h/pmm2h)
    #     bias_ps = pge2h2/pmm2h
    #     bias_theo = halomodel.galaxybias(zs,log10mlow,log10mhigh)[z_id]
    #     print (bias_ps, bias_theo, bias_ps/bias_theo, bias_ps/bias_theo)
    # #return
    

    #-------------calc and plot power spectra
    log10mlow= 13.4
    log10mhigh= 13.9
    plt.rcParams.update({'font.size': 14})
    
    # plt.loglog(ks,(ks**3)*(halomodel.P_mm_1h(ks,zs)[z_id,:])/(2.*np.pi**2),color='k')
    # plt.loglog(ks,(ks**3)*(halomodel.P_mm_1h(ks,zs,log10mlow,log10mhigh)[z_id,:])/(2.*np.pi**2),color='k',linestyle='--')
    # plt.loglog(ks,(ks**3)*(halomodel.P_mm_2h(ks,zs)[z_id,:])/(2.*np.pi**2),color='k')
    # plt.loglog(ks,(ks**3)*(halomodel.P_mm_2h(ks,zs,log10mlow,log10mhigh)[z_id,:])/(2.*np.pi**2),color='k',linestyle='--')

    # plt.loglog(ks,(ks**3)*(halomodel.P_gg_1h(ks,zs)[z_id,:])/(2.*np.pi**2),color='g')
    # plt.loglog(ks,(ks**3)*(halomodel.P_gg_1h(ks,zs,log10mlow,log10mhigh)[z_id,:])/(2.*np.pi**2),color='g',linestyle='--')
    # plt.loglog(ks,(ks**3)*(halomodel.P_gg_2h(ks,zs)[z_id,:])/(2.*np.pi**2),color='g')
    # plt.loglog(ks,(ks**3)*(halomodel.P_gg_2h(ks,zs,log10mlow,log10mhigh)[z_id,:])/(2.*np.pi**2),color='g',linestyle='--')   

    # plt.loglog(ks,(ks**3)*(halomodel.P_ee_1h(ks,zs)[z_id,:])/(2.*np.pi**2),color='r')
    # plt.loglog(ks,(ks**3)*(halomodel.P_ee_1h(ks,zs,log10mlow=log10mlow,log10mhigh=log10mhigh)[z_id,:])/(2.*np.pi**2),color='r',linestyle='--')
    # plt.loglog(ks,(ks**3)*(halomodel.P_ee_2h(ks,zs)[z_id,:])/(2.*np.pi**2),color='r')
    # plt.loglog(ks,(ks**3)*(halomodel.P_ee_2h(ks,zs,log10mlow=log10mlow,log10mhigh=log10mhigh)[z_id,:])/(2.*np.pi**2),color='r',linestyle='--')   

    # plt.loglog(ks,(ks**3)*(halomodel.P_ge_1h(ks,zs)[z_id,:])/(2.*np.pi**2),color='y')
    # plt.loglog(ks,(ks**3)*(halomodel.P_ge_1h(ks,zs,log10mlow=log10mlow,log10mhigh=log10mhigh)[z_id,:])/(2.*np.pi**2),color='y',linestyle='--')
    # plt.loglog(ks,(ks**3)*(halomodel.P_ge_2h(ks,zs)[z_id,:])/(2.*np.pi**2),color='y')
    # plt.loglog(ks,(ks**3)*(halomodel.P_ge_2h(ks,zs,log10mlow=log10mlow,log10mhigh=log10mhigh)[z_id,:])/(2.*np.pi**2),color='y',linestyle='--')   
    
    # plt.loglog(ks,(ks**3)*halomodel.pk[z_id,:]/(2.*np.pi**2),color='b')
    #plt.loglog(ks,(ks**3)*halomodel.pknl[z_id,:]/(2.*np.pi**2),color='b')

    #compare matt
    #plt.loglog(ks,(ks**3)*(halomodel.P_ee_1h(ks,zs)[z_id,:])/(2.*np.pi**2),color='r')
    #plt.loglog(ks,(ks**3)*(halomodel.P_ee_2h(ks,zs)[z_id,:])/(2.*np.pi**2),color='g')

    # plt.loglog(ks,(ks**3)*(halomodel.P_ee_1h(ks,zs,gasprofile='AGN')[z_id,:]+halomodel.P_ee_2h(ks,zs,gasprofile='AGN')[z_id,:])/(2.*np.pi**2),color='blue')
    # plt.loglog(ks,(ks**3)*(halomodel.P_gg_1h(ks,zs)[z_id,:]+halomodel.P_gg_2h(ks,zs)[z_id,:])/(2.*np.pi**2),color='black')
    # plt.loglog(ks,(ks**3)*(halomodel.P_ge_1h(ks,zs,gasprofile='AGN')[z_id,:]+halomodel.P_ge_2h(ks,zs,gasprofile='AGN')[z_id,:])/(2.*np.pi**2),color='red')
    
    pee1h = halomodel.P_ee_1h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pee2h = halomodel.P_ee_2h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pgg1h = halomodel.P_gg_1h(ks,zs,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pgg2h = halomodel.P_gg_2h(ks,zs,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pmm1h = halomodel.P_mm_1h(ks,zs,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pmm2h = halomodel.P_mm_2h(ks,zs,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pge1h = halomodel.P_ge_1h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pge2h = halomodel.P_ge_2h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pme1h = halomodel.P_me_1h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    pme2h = halomodel.P_me_2h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    phe1h = halomodel.P_he_1h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    phe2h = halomodel.P_he_2h(ks,zs,gasprofile='AGN',logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    phh1h = halomodel.P_hh_1h(ks,zs,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    phh2h = halomodel.P_hh_2h(ks,zs,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    
    #plt.loglog(ks,(ks**3)*(pee1h+pee2h)/(2.*np.pi**2),color='blue',label="Pee")
    #plt.loglog(ks,(ks**3)*(pgg1h+pgg2h)/(2.*np.pi**2),color='black',label="Pgg")
    #plt.loglog(ks,(ks**3)*(pmm1h+pmm2h)/(2.*np.pi**2),color='black',label="Pmm",ls="dashed")    
    #plt.loglog(ks,(ks**3)*(pge1h+pge2h)/(2.*np.pi**2),color='red',label="Pge")
    #plt.loglog(ks,(ks**3)*(pge1h+pge2h)/(2.*np.pi**2),color='red',label="Pge_all_e")
    #plt.loglog(ks,(ks**3)*(pme1h+pme2h)/(2.*np.pi**2),color='red',ls='dashed',label="Pme_all_e")

    plt.loglog(ks,(ks**3)*(pge1h+pge2h)**2./(pgg1h+pgg2h)/(2.*np.pi**2),color='red',label="Pge_all_e^2/Pgg")
    plt.loglog(ks,(ks**3)*(phe1h+phe2h)**2./(phh1h+phh2h)/(2.*np.pi**2),color='black',ls='dashed',label="Phe_all_e^2/Phh")
    
    #plt.loglog(ks,(ks**3)*(pgg1h+pgg2h2)/(2.*np.pi**2),color='black',ls='dashed',label="Pgg cross")
    
    #io.save_cols("test2.txt",(ks,pee1h,pee2h,pgg1h,pgg2h,pge1h,pge2h))
    # plt.loglog(ks,(pgg1h+pgg2h)/(2.*np.pi**2),color='red',label="agn Pgg")
    # kstar = 0.03
    # pgg1h_decay = pgg1h*(1.-np.exp(-(ks/kstar)**2.))
    # plt.loglog(ks,(pgg1h_decay+pgg2h)/(2.*np.pi**2),color='blue',ls='dashed',label="agn Pgg damped")
    
    #plt.loglog(ks,(ks**3)*(pge1h+pge2h)/(2.*np.pi**2),color='red',label="agn Pge")
    #galaxybias = halomodel.galaxybias(zs,log10mlow,log10mhigh)[z_id]
    #plt.loglog(halomodel.k,galaxybias*(halomodel.k**3)*halomodel.pknl[z_id,:]/(2.*np.pi**2),color='b',label="camb x bias")
    
    #plt.loglog(ks,(ks**3)*(pge1h2+pge2h2)/(2.*np.pi**2),color='blue')  
    #plt.xlim([.1,10.])
    #plt.ylim([.1,10000])
    plt.grid()
    #plt.ylabel(r'$k^3 P(k)/2\pi^2$')
    plt.ylabel(r'$P(k)/2\pi^2$')
    plt.xlabel(r'$k$')
    plt.tight_layout()
    plt.legend(loc=1,frameon=False)
    plt.show()
    plt.savefig('plots/halo_powerspectra_2.png')


    #-------------check number densities
    log10mlow=13.
    log10mhigh=16.
    z = 0.001 #1.5
    ngalMpc3 = halomodel.nbar_halo(z,log10mlow,log10mhigh)[0]
    galaxybias = halomodel.bias_galaxy(z,log10mlow,log10mhigh)[0]
    print ("ngalMpc3", ngalMpc3)
    print ("galaxybias", galaxybias)   
    
    
   


if (__name__ == "__main__"):
    main(sys.argv[1:])




