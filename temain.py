import os
import platform
import ctypes

import numpy as np
import pandas as pd

# From Python 3.8 onwards, there is a reported bug in CDLL.__init__()
mode = dict(winmode=1) if platform.python_version() >= '3.8' else dict()
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = ctypes.CDLL(os.path.join(script_dir, 'libtemain.so'), **mode)

#########################
libc = ctypes.CDLL("libc.so.6")  # On Linux
libc.fflush.argtypes = [ctypes.c_void_p]
libc.fflush.restype = ctypes.c_int
#########################

# Subclassing the pointer to be able to pass None
def ndpointer(dtype, ndim, flags='C_CONTIGUOUS'):
    _ComplexArrayTypeBase = np.ctypeslib.ndpointer(dtype=dtype, ndim=ndim, flags=flags)
    def _from_param(cls, obj):
        if obj is None:
            return obj
        return _ComplexArrayTypeBase.from_param(obj)
    ComplexArrayType = type(
        'ComplexArrayType',
        (_ComplexArrayTypeBase,),
        {'from_param': classmethod(_from_param)}
    )
    return ComplexArrayType

# temain::Temain* temain_init(double seed);
lib.temain_init.argtypes = [ctypes.c_double]
lib.temain_init.restype = ctypes.c_void_p
# temain::Temain * temain_copy(temain::Temain * self)
lib.temain_copy.argtypes = [ctypes.c_void_p]
lib.temain_copy.restype = ctypes.c_void_p
# void temain_simulate(temain::Temain * self, long steps, double * buffer, double * mv, double * dv);
lib.temain_simulate.argtypes = [ctypes.c_void_p, ctypes.c_long, 
                                ndpointer(dtype=np.float64, ndim=2), ndpointer(dtype=np.float64, ndim=2),
                                ndpointer(dtype=bool, ndim=2)]
# void temain_free(temain::Temain * self) { delete self; }
lib.temain_free.argtypes = [ctypes.c_void_p]
# void temain_getYP(temain::Temain * self, double * buffer) { self->getYP(buffer); }
lib.temain_getYP.argtypes = [ctypes.c_void_p, ndpointer(dtype=np.float64, ndim=1)]
# void temain_jacobian(temain::Temain * self, double* buffer) { self->jacobian(buffer); }
lib.temain_jacobian.argtypes = [ctypes.c_void_p, ndpointer(dtype=np.float64, ndim=2)]

NMV = 12
NCV = 41
FREQ = "1s"

# A double of the C++ Temain class
class BaseTemain(object):
    def __init__(self, seed=1, obj=None):
        assert seed > 0 or seed < 0, "Seed must be greater than 1"
        self.obj = lib.temain_init(seed) if obj is None else obj
        self.columns = self.info().index

    def norm(self, return_YP=False):
        YP = np.zeros(50)
        lib.temain_getYP(self.obj, YP)
        if return_YP:
            return YP
        return np.linalg.norm(YP[:38])
    
    def clone(self):
        return self.__class__(obj=lib.temain_copy(self.obj))
    
    def __del__(self):
        lib.temain_free(self.obj)
    
    def simulate(self, n=None, MV=None, DV=None, keep_provided=False):
        if n is None:
            if MV is None and DV is None: n = 1
            if MV is not None: n = len(MV)
            if DV is not None: n = len(DV)
        else: assert n >= 0
        if MV is not None:
            assert len(MV) == n
            MV = np.asarray(MV).astype("float64")
        if DV is not None:
            assert len(DV) == n
            DV = np.asarray(DV).astype(bool)
        buffer = np.zeros((n, NMV+NCV), dtype="float64")
        lib.temain_simulate(self.obj, n, buffer, MV, DV)
        if not keep_provided and MV is not None:
            buffer[:, -NMV:] = MV
        index = np.arange(0, n, 1, dtype='datetime64[s]')
        return pd.DataFrame(data=buffer, index=index, columns=self.columns)
    
    @staticmethod
    def info():
        description = ['A Feed (stream 1)', 'D Feed (stream 2)', 'E Feed (stream 3)', 'A and C Feed (stream 4)',
                       'Recycle Flow (stream 8)', 'Reactor Feed Rate (stream 6)', 'Reactor Pressure', 'Reactor Level',
                       'Reactor Temperature', 'Purge Rate (stream 9)', 'Product Sep Temp', 'Product Sep Level',
                       'Prod Sep Pressure', 'Prod Sep Underflow (stream 10)', 'Stripper Level', 'Stripper Pressure',
                       'Stripper Underflow (stream 11)', 'Stripper Temperature', 'Stripper Steam Flow', 'Compressor Work',
                       'Reactor Cooling Water Outlet Temp', 'Separator Cooling Water Outlet Temp', 'Component A (stream 6)',
                       'Component B (stream 6)', 'Component C (stream 6)', 'Component D (stream 6)', 'Component E (stream 6)',
                       'Component F (stream 6)', 'Component A (stream 9)', 'Component B (stream 9)', 'Component C (stream 9)',
                       'Component D (stream 9)', 'Component E (stream 9)', 'Component F (stream 9)', 'Component G (stream 9)',
                       'Component H (stream 9)', 'Component D (stream 11)', 'Component E (stream 11)', 'Component F (stream 11)',
                       'Component G (stream 11)', 'Component H (stream 11)', 'D Feed Flow (stream 2)', 'E Feed Flow (stream 3)',
                       'A Feed Flow (stream 1)', 'A and C Feed Flow (stream 4)', 'Compressor Recycle Valve', 'Purge Valve (stream 9)',
                       'Separator Pot Liquid Flow (stream 10)', 'Stripper Liquid Product Flow (stream 11)', 'Stripper Steam Valve',
                       'Reactor Cooling Water Flow', 'Condenser Cooling Water Flow', 'Agitator Speed']
        unit = ['kscmh', 'kg/h', 'kg/h', 'kscmh', 'kscmh', 'kscmh', 'kPa', '%', 'oC', 'kscmh', 'oC', '%', 'kPa', 'm^3/h',
                '%', 'kPa', 'm^3/h', 'oC', 'kg/h', 'kW', 'oC', 'oC', *['mole %' for i in range(19)], *['%' for i in range(NMV)]]
        variable = ["CV[{:}]".format(i) for i in range(0, NCV)] + ["MV[{:}]".format(i) for i in range(0, NMV)]
        info = pd.DataFrame(data={'description': description, 'unit': unit}, index=variable)
        # soft limits
        info.lower = np.nan
        info.loc[[f"CV[{i}]" for i in (7, 11)], "lower"] = 50, 30
        info.loc[[f"MV[{i}]" for i in range(12)], "lower"] = 0
        info.upper = np.nan
        info.loc[[f"CV[{i}]" for i in (6, 7, 8, 11)], "upper"] = 2895, 100, 150, 100
        info.loc[[f"MV[{i}]" for i in range(12)], "upper"] = 100
        return info

class TemainProcess(BaseTemain):
    def __init__(self, freq="1s", **kwargs):
        self.freq = freq
        self.ratio = pd.Timedelta(self.freq) // pd.Timedelta(FREQ)
        super().__init__(**kwargs)
    
    def clone(self):
        return self.__class__(self.freq, obj=lib.temain_copy(self.obj))
    
    def simulate(self, n=None, MV=None, DV=None, resample="mean", interpolate="ffill", **kwargs):
        if n is not None: n *= self.ratio
        if interpolate == "ffill":
            if MV is not None: MV = np.repeat(MV, axis=0, repeats=self.ratio)
            if DV is not None: DV = np.repeat(DV, axis=0, repeats=self.ratio)
        else:
            raise NotImplementedError()
        #return getattr(super().simulate(n=n, MV=MV, DV=DV, **kwargs).resample(self.freq), resample)()
        result = getattr(super().simulate(n=n, MV=MV, DV=DV, **kwargs).resample(self.freq), resample)()
        libc.fflush(None)
        return result

class TemainAnalyser:
    
    COMPRESSOR_COST = 0.0536
    STEAM_COST = 0.0318
    PRODUCT_MOLAR_FLOW = 9.20697
    PURGE_MOLAR_FLOW = 44.79
    
    def __init__(self, freq="60s", f=lambda x: x):
        self.ratio = pd.Timedelta(freq) / pd.Timedelta("60s")
        self.f = f
        self._init_cost()
    
    @staticmethod
    def _to_values(data):
        return data.values if type(data) == pd.DataFrame else data
    
    @staticmethod
    def component_cost():
        return pd.Series(index=["A", "B", "C", "D", "E", "F", "G", "H"], 
                         data=[2.206, 0, 6.177, 22.06, 14.56, 17.89, 30.44, 22.94],
                         name="Component cost, $/kgmol")
    
    @staticmethod
    def component_density():
        return pd.Series(data=[299, 365, 328, 612, 617], 
                         index=["D", "E", "F", "G", "H"], 
                         name="Liquid Density (at 100 degrees), kg/m^3")
    
    @staticmethod
    def component_molar():
        return pd.Series(data=[2, 25.4, 28.0, 32.0, 46.0, 48.0, 62.0, 76.0], 
                         index=["A", "B", "C", "D", "E", "F", "G", "H"], 
                         name="Molecular Weight")

    #@staticmethod
    #def purge_fraction(): # Unused
    #    return pd.Series(data=[0.32958, 0, 0.23978, 0.01257, 0.18579, 0.02263, 0.04844, 0.02299], 
    #                     index=["A", "B", "C", "D", "E", "F", "G", "H"], 
    #                     name="Purge Mole Fraction")

    #@staticmethod
    #def loss_fraction(): # Unused
    #    return pd.Series(data=[0, 0, 0, 0.00018, 0.00836, 0.00099, 0, 0], 
    #                     index=["A", "B", "C", "D", "E", "F", "G", "H"], 
    #                     name="Product Losses Mole Fraction")
    
    def _init_cost(self):
        self._component_cost = self.f(self.component_cost().values)
        self._component_density = self.f(self.component_density().values)
        self._component_molar = self.f(self.component_molar().values)
        
    def purge_losses(self, data):
        X = self._to_values(data)
        kgmol_purge = (X[:, 28:36] / 100 * self._component_cost[None]).sum(axis=1)
        purge_rate = X[:, 9]
        molar_flow = 44.79
        return kgmol_purge * molar_flow * purge_rate * self.ratio
    
    # Stream 11, kgmol/h
    #def product_flow(self, data):
    #    X = self._to_values(data)
    #    vol = X[:, 16]
    #    mole_ratio = X[:, 36:41] / 100
    #    molar = self._component_molar[None, -5:]
    #    return vol / (molar * mole_ratio / self._component_density).sum(axis=1)
    
    def product_losses(self, data):
        X = self._to_values(data)
        # 36:39 - D, E, F. Component cost 3:6 - D, E, F. G, H are zeros anyway 
        kgmol_product = (X[:, 36:39] / 100 * self._component_cost[None, 3:6]).sum(axis=1)
        product_rate = X[:, 16]
        return kgmol_product * self.PRODUCT_MOLAR_FLOW * product_rate * self.ratio
    
    def compressor_losses(self, data):
        X = self._to_values(data)
        compressor_work = X[:, 19]
        return self.COMPRESSOR_COST * compressor_work * self.ratio
    
    def steam_losses(self, data):
        X = self._to_values(data)
        steam_flow = X[:, 18]
        return self.STEAM_COST * steam_flow * self.ratio
    
    def __call__(self, data):
        X = self._to_values(data)
        shape = X.shape
        assert len(X.shape) < 4
        if len(shape) > 2:
            X = X.reshape(-1, X.shape[-1])
        cost = self.cost(X)
        if len(shape) > 2:
            cost = cost.reshape(shape[:2] + (1,))
        return cost
    
    def cost(self, data):
        return self.purge_losses(data) + self.product_losses(data) +\
            self.compressor_losses(data) + self.steam_losses(data)
    
    #def product_ratio(self, data):
    #    X = self._to_values(data)
    #    products = X[:, 39:41] * self._component_molar[-2:]
    #    return products[:, 0] / products[:, 1]