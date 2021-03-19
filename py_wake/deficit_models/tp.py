from numpy import newaxis as na

import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.utils.area_overlapping_factor import AreaOverlappingFactor
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter
from py_wake.turbulence_models.stf import STF2017TurbulenceModel

class TPDeficit(NOJDeficit, AreaOverlappingFactor):
    """
    Largely identical to NOJDeficit(), however using local quantities for the
    inflow wind speed and turbulence intensity. The latter input is a also a new
    addition as the wake expansion factor, k, is now a function of the local
    TI. The relationship between TI and k is taken from the linear connection
    Niayifar and Porte-Agel (2016) estbalished for the Gaussian wake model.
    The expansion rates in the Jensen and Gaussian describe the same process.
    """
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'D_src_il', 'D_dst_ijl', 'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'TI_eff_ilk', 'TI_ilk']

    def __init__(self, k=0.6, use_effective_ws=True):
        self.k = k
        self.use_effective_ws = use_effective_ws

    def _calc_layout_terms(self, WS_ilk, WS_eff_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        WS_ref_ilk = (WS_ilk, WS_eff_ilk)[self.use_effective_ws]
        R_src_il = D_src_il / 2
        wake_radius = self._wake_radius(WS_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs)
        term_denominator_ijlk = (wake_radius / R_src_il[:, na, :, na])**2
        term_denominator_ijlk += (term_denominator_ijlk == 0)
        A_ol_factor_ijlk = self.overlapping_area_factor(wake_radius, dw_ijlk, cw_ijlk, D_src_il, D_dst_ijl)

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            self.layout_factor_ijlk = WS_ref_ilk[:, na] * (dw_ijlk > 0) * (A_ol_factor_ijlk / term_denominator_ijlk)

    def calc_deficit(self, WS_ilk, WS_eff_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        if not self.deficit_initalized:
            self._calc_layout_terms(WS_ilk, WS_eff_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk, ct_ilk, **kwargs)
        ct_ilk = np.minimum(ct_ilk, 1)   # treat ct_ilk for np.sqrt()
        term_numerator_ilk = (1 - np.sqrt(1 - ct_ilk))
        return term_numerator_ilk[:, na] * self.layout_factor_ijlk

    def _calcTI(self,ws,z=140,wti_fac=1.):
        (a1,a2,a3)=(0.035,0.0089,0.0402)
        zref=10.
        Uref=10.
        TI=(a1*(ws/Uref)+a2+a3*(1/(ws/Uref)))*(z/zref)**(-0.22)
        return TI*wti_fac

    def _wake_radius(self, WS_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        k_ijlk = np.atleast_3d(self.k)[:, na]
        D=D_src_il[:, na, :, na]
        #TI=kwargs.get('TI_ilk', 0)[:, na]
        TI=kwargs.get('TI_ilk', 0)
        if TI.all()==0:
            TI=self._calcTI(WS_ilk)[:, na]
        else:
            TI=TI[:, na]
        ct = np.minimum(ct_ilk, 1)   # treat ct_ilk for np.sqrt()
        alp=1.5*TI
        bet=((0.8*TI)/np.sqrt(ct)[:, na])
        d=dw_ijlk*(dw_ijlk > 0)
        wake_radius_ijlk = 0.5*D + (0.5/bet)*k_ijlk*TI*D * (
            np.sqrt( ( alp + bet*(d/D) )**2 + 1 ) - np.sqrt( 1 + alp**2 ) 
            - np.log( ( ( np.sqrt( (alp+bet*(d/D))**2 + 1 ) + 1 )*alp ) / ( ( np.sqrt( 1 + alp**2 ) + 1 ) * ( alp + bet*(d/D) ) ) )
        )
        #print(wake_radius_ijlk.min(), wake_radius_ijlk.mean(), wake_radius_ijlk.max())
        return wake_radius_ijlk

    def wake_radius(self, WS_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        return self._wake_radius(WS_ilk, D_src_il, dw_ijlk, ct_ilk, **kwargs)[0]



class TP(PropagateDownwind):
    def __init__(self, site, windTurbines, rotorAvgModel=RotorCenter(),
                 k=0.6, use_effective_ws=True,
                 superpositionModel=LinearSum(),
                 #superpositionModel=SquaredSum(),
                 deflectionModel=None,
                 turbulenceModel=STF2017TurbulenceModel()):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float, default 0.1
            wake expansion factor
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel, default None
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=TPDeficit(k=k, use_effective_ws=use_effective_ws),
                                   rotorAvgModel=rotorAvgModel,
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)