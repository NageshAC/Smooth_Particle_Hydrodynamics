import ctypes as Ct

dllHandle = Ct.CDLL(r"D:/Projects/Smooth Particle Hydrodynamics/test/test_dll.dll")

ADD = dllHandle.add
ADD.restype = Ct.c_double
ADD.argtypes = [Ct.POINTER(Ct.c_double), Ct.POINTER(Ct.c_double)]
a1 = Ct.c_double(1)
a2 = Ct.c_double(3.2)
print(ADD(Ct.byref(a1),Ct.byref(a2)))
# print(ADD(a1,a2))