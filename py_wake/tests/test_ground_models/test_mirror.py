from py_wake.site._site import UniformSite
from py_wake.examples.data.hornsrev1 import V80
from py_wake.ground_models import Mirror
from py_wake.deficit_models.noj import NOJ, NOJDeficit
import matplotlib.pyplot as plt
from py_wake.flow_map import YZGrid
import numpy as np
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines, OneTypeWindTurbines
from py_wake.superposition_models import LinearSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
import pytest
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, IEA37SimpleBastankhahGaussianDeficit,\
    ZongGaussianDeficit, NiayifarGaussianDeficit
from py_wake.deficit_models.fuga import FugaDeficit
from py_wake.deficit_models.gcl import GCLDeficit
from py_wake.turbulence_models.stf import STF2017TurbulenceModel


def test_Mirror_NOJ():
    site = UniformSite([1], ti=0.1)
    V80_D0 = V80()
    V80_D0._diameters = [0]
    wt = WindTurbines.from_WindTurbines([V80(), V80_D0])
    wfm = NOJ(site, wt, k=.5, groundModel=Mirror())
    sim_res = wfm([0], [0], h=[50], wd=0)
    fm_ref = sim_res.flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10))
    ref = fm_ref.WS_eff_xylk[:, 0, 0, 0].values

    res = np.array([wfm([0, 0], [0, y], [50, 10], type=[0, 1], wd=0).WS_eff.sel(wt=1).item()
                    for y in fm_ref.X[0]])

    if 0:
        fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 10, 1)))
        fm_res.plot_wake_map()
        plt.plot(fm_ref.X[0], fm_ref.Y[0], '.')
        plt.plot(fm_ref.X[0], ref * 10, label='ref, WS*10')
        plt.plot(fm_ref.X[0], res * 10, label='Res, WS*10')

        plt.legend()
        plt.show()
    plt.close()
    npt.assert_array_equal(res, ref)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
def test_Mirror(wfm_cls):
    site = UniformSite([1], ti=0.1)
    wt = V80()
    wfm = wfm_cls(site, wt, ZongGaussianDeficit(a=[0, 1]),
                  turbulenceModel=STF2017TurbulenceModel(), groundModel=Mirror())
    sim_res = wfm([0], [0], h=[50], wd=0,)
    fm_ref = sim_res.flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10))
    ref = fm_ref.WS_eff_xylk[:, 0, 0, 0].values

    res = np.array([wfm([0, 0], [0, y], [50, 10], wd=0).WS_eff.sel(wt=1).item()
                    for y in fm_ref.X[0]])

    if 0:
        fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 10, 1)))
        fm_res.plot_wake_map()
        plt.plot(fm_ref.X[0], fm_ref.Y[0], '.')
        plt.plot(fm_ref.X[0], ref * 10, label='ref, WS*10')
        plt.plot(fm_ref.X[0], res * 10, label='Res, WS*10')

        plt.legend()
        plt.show()
    plt.close()
    npt.assert_array_equal(res, ref)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
def test_Mirror_flow_map(wfm_cls):
    site = UniformSite([1], ti=0.1)
    wt = V80()
    wfm = NOJ(site, wt, k=.5, superpositionModel=LinearSum())

    fm_ref = wfm([0, 0 + 1e-20], [0, 0 + 1e-20], wd=0, h=[50, -50]
                 ).flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)))
    fm_ref.plot_wake_map()
    plt.title("Underground WT added manually")

    plt.figure()
    wfm = wfm_cls(site, wt, NOJDeficit(k=.5), groundModel=Mirror())
    fm_res = wfm([0], [0], wd=0, h=[50]).flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)))
    fm_res.plot_wake_map()
    plt.title("With Mirror GroundModel")

    if 0:
        plt.show()
    plt.close()
    npt.assert_array_equal(fm_ref.WS_eff, fm_res.WS_eff)