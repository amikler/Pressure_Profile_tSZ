import pdb

def cosmo_cal (z, H_0=70, omega_M = 0.3, omega_lambda = 0.7, omega_k = 0, h_f=100): #H_0=None
    """
    cosmo= cosmo_cal (z,H_0=None)
    
    A function to calculate the different cosmological distances using LCDM cosmology.
    
    :INPUT:
      z  - redishift of the cluster.

      H_0 - Hublle constant by default is 70
    
    :OUTPUT:
        cosmo- a data structure containing the following
                   distance calculations, computed at the given redshift:

          .dh     - 

          .dc  - comoving distance

          .dm   - 

          .da    - angular distance

          .ez - 


    :EXAMPLE:        
      ::
        
        import cosmo_cal as cc

        cosmo_dist = cc.cosmo_cal(z, H_0=69)

        ang_dist=cosmo_dist.da
    """
    

 
    import numpy as np
    from scipy.integrate import quad

    class cosmoCal:
        """Empty object container.
        """
        def __init__(self): 
            self.dh = None
            self.dm= None
            self.da = None
            self.dc = None
            self.ez = None
            

    #---------------------
    # Set up input parameters
    #---------------------
    z_cluster = z
    
    # if H_0==None:
    #     H_0 = 70 #h Km s-1 Mpc -1

    # omega_M = 0.3
    # omega_lambda = 0.7
    # omega_k = 0
    cl=2.9979246*10**8 #m/s^2
    h=H_0/h_f#100.
    
    E_z= np.sqrt(omega_M*(1+z_cluster)**3 +omega_k*(1+z_cluster)**2 +omega_lambda)

    integral, extra = quad(lambda z:(1/(np.sqrt((omega_M*(1+z)**3) +(omega_k*(1+z)**2)+omega_lambda))), 0, z_cluster)
    D_H = (cl/1000)/H_0
    D_C = D_H *integral #Mpc 
    D_M = D_C #for omega_k = 0 Mpc
    D_A = D_M/(1+z_cluster)


    #---------------------
    # Prepare the data container 
    #---------------------
    cosmocal =  cosmoCal()

    cosmocal.dh= D_H
    cosmocal.dc=D_C
    cosmocal.dm=D_M
    cosmocal.da=D_A
    cosmocal.ez=E_z
   
    #---------------------
    # Return with data
    #---------------------
    
    return cosmocal

