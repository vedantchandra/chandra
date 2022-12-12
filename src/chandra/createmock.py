
import socket
hostname = socket.gethostname()
if hostname[:4] == 'holy':
    default_specNN = '/n/holyscratch01/conroy_lab/pacargile/ThePayne/Hecto_FAL/lowres/YSTANN_4000_7000_spec.h5' #not there right now, folder empty
    default_contNN = '/n/holyscratch01/conroy_lab/pacargile/ThePayne/Hecto_FAL/lowres/YSTANN_4000_7000_cont.h5'
    default_photNN = '/n/holyscratch01/conroy_lab/pacargile/ThePayne/SED/VARRV/'
    default_MISTgrid = '/n/holyscratch01/conroy_lab/pacargile/MIST/MIST_2.0_spot_EEPtrk_small.h5'
else:
    # change the following to where ever you store the photometric NNs and the MIST tracks
    default_specNN = '/Users/vedantchandra/0_research/00_outerhalo/03_gaia_bprp/MSfiles/YSTANN_4000_7000_spec.h5'
    default_contNN = '/Users/vedantchandra/0_research/00_outerhalo/03_gaia_bprp/MSfiles/YSTANN_4000_7000_cont.h5'
    default_photNN = '/Users/vedantchandra/0_research/software/MS_files/VARRV/'
    default_MISTgrid = '/Users/vedantchandra/0_research/software/MS_files/MIST_2.0_spot_EEPtrk_small.h5'

default_filtarr = ([
    'GaiaEDR3_BP','GaiaEDR3_G','GaiaEDR3_RP',
    'PS_g','PS_r','PS_i','PS_z',
    '2MASS_H','2MASS_J','2MASS_Ks',
    'WISE_W1','WISE_W2', 'VISTA_J', 'VISTA_H', 'VISTA_Ks'])

import numpy as np
from minesweeper.fastMISTmod import GenMIST
from Payne.fitting.genmod import GenMod as GMOD

print('lmao')

class genmock(object):
    """
    
    class to simplify building mock data directly from MIST and The Payne models.

    Parameters
    ----------
        
        specNN : str
            Path to spectral ANN.
        
        photNN : str
            Path to directory containing all of the 
            photometric ANN files.

        MISTgrid : str
            Path to MIST eeptrack file.

        specbool : bool
            Flag used to turn on/off spectral predictons.

        photbool : bool
            Flag used to turn on/off photometric predictions.

        filters : list
            List of filter names to include in phot predictions. 
            This has no effect if photbool == False.

    """
    def __init__(self, *arg, **kwargs):
        super(genmock, self).__init__()

        self.specNN = kwargs.get('specNN',default_specNN)
        self.contNN = kwargs.get('contNN',default_contNN)
        self.photNN = kwargs.get('photNN',default_photNN)
        self.MISTgrid = kwargs.get('MISTgrid',default_MISTgrid)
        
        self.specbool = kwargs.get('specbool',True)
        self.photbool = kwargs.get('photbool',True)

        self.filtarr = kwargs.get('filters',default_filtarr)

        GM = GMOD()

        GMIST = GenMIST(MISTpath=self.MISTgrid)
        self.genMISTfn = GMIST.getMIST
        self.mistpars = GMIST.modpararr
        
        if self.specbool:
            GM._initspecnn(nnpath=self.specNN,NNtype='LinNet',Cnnpath=self.contNN)
            self.genspecfn = GM.genspec
        
        if self.photbool:
            GM._initphotnn(self.filtarr,nnpath=self.photNN)
            self.genphotfn = GM.genphot

    def run(self,**indict):
        """
        
        Function to generate mock spectra and photometry. The specific 
        input parameters needed for the function depend on specbool and 
        photbool, i.e., if the code is predicting spectra and/or photometry.

        Parameters
        ----------
            
            eep : float
                input EEP

            initial_Mass : float
                input initial mass in solar masses

            initial_[Fe/H] : float
                input initial [Fe/H]

            initial_[a/Fe] : float
                input initial [a/Fe]

            dist : float
                input distance in pc

            av : float
                input Av

            vrad : float
                input radial velocity in km/s

            vstar : float
                input stellar broadening in km/s (i.e., vsini + vmacro)

            pc0 : float
                input 0th term in Cheb

            pc1 : float
                input 1st term in Cheb

            pc2 : float
                input 2nd term in Cheb

            pc3 : float
                input 3rd term in Cheb

            instr : float
                input instrument resolution

        Returns
        -------
        
            output : dict
                Returns a dictionary containing three keys: 'spec', 'phot',
                and 'mist'. 'mist' contains all of the MIST predictions. 'spec'
                contains a dictionary with 'wave' and 'flux'. 'phot' contains 
                a dictionary with the predicted photometry for all of the different 
                filters in the input filter array.

        """

        # pull out stellar paramters, anything not included will get
        # some default value
        eep       = indict.get('eep',np.nan)
        init_mass = indict.get('initial_Mass',np.nan)
        init_feh  = indict.get('initial_[Fe/H]',np.nan)
        init_afe  = indict.get('initial_[a/Fe]',np.nan)
        dist      = indict.get('dist',np.nan)
        av        = indict.get('av',np.nan)
        vrad      = indict.get('vrad',np.nan)
        vstar     = indict.get('vstellar',np.nan)
        pc0       = indict.get('pc0',np.nan)
        pc1       = indict.get('pc1',np.nan)
        pc2       = indict.get('pc2',np.nan)
        pc3       = indict.get('pc3',np.nan)
        instr     = indict.get('instr',np.nan)

        # first do the MIST prediction
        MISTpred = self.genMISTfn(
            eep=eep,
            mass=init_mass,
            feh=init_feh,
            afe=init_afe,
            )
        MISTdict = ({
            kk:pp for kk,pp in zip(
            self.mistpars,MISTpred)
            })

        if self.specbool:
            teff = 10.0**MISTdict['log(Teff)']
            logg = MISTdict['log(g)']
            feh  = MISTdict['[Fe/H]']
            afe  = MISTdict['[a/Fe]']

            specpars = [teff,logg,feh,afe,vrad,vstar,np.nan,instr]
            specpars += [pc0,pc1,pc2,pc3]

            specwave_in,specflux_in = self.genspecfn(specpars,modpoly=True)

            specout = {'wave':specwave_in,'flux':specflux_in}
        else:
            specout = None

        if self.photbool:
            teff = 10.0**MISTdict['log(Teff)']
            logg = MISTdict['log(g)']
            feh  = MISTdict['[Fe/H]']
            afe  = MISTdict['[a/Fe]']
            logr = MISTdict['log(R)']

            photpars = [teff,logg,feh,afe,logr,dist,av,3.1]
            photout = self.genphotfn(photpars)
        else:
            photout = None

        return {'spec':specout,'phot':photout,'mist':MISTdict}

if __name__ == '__main__':
    mockclass = genmock()

    inputdict = ({
        'eep':350.0,
        'initial_Mass':1.0,
        'initial_[Fe/H]':0.0,
        'initial_[a/Fe]':0.0,
        'dist':100.0,
        'av':0.5,
        'vrad':-5.0,
        'vstellar':4.0,
        'pc0':1.0,
        'pc1':0.0,
        'pc2':0.0,
        'pc3':0.0,
        'instr':2000.0,
        })

    output = mockclass.run(**inputdict)

    print(output)