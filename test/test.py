import unittest as UT
import ctypes as Ct
from pathlib import Path

class test_SPH(UT.TestCase):
    def setUp(self):
        self.dllPath = Path(__file__).parents[1] / "build"

        # importing and setting up Smoothing Kernels
        self.SKdllHandle = Ct.CDLL(str(self.dllPath / "smoothing_kernels.dll"))
        self.W_poly6 = self.SKdllHandle.p_W_poly6
        self.W_poly6.restype = Ct.c_double
        self.W_poly6.argtypes = [Ct.POINTER(Ct.c_double), Ct.POINTER(Ct.c_double), Ct.POINTER(Ct.c_double)] 

    def tearDown(self):
        pass

    def test_W_poly6(self):

        r2      = Ct.c_double(1.)
        C_h2    = Ct.c_double(1. ** 2.)
        C_1_h9  = Ct.c_double(1. ** 9.)

        self.assertEqual(self.W_poly6(Ct.byref(r2), Ct.byref(C_h2), Ct.byref(C_1_h9)), 0.)


if __name__ == '__main__':
    UT.main()