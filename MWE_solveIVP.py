import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import scipy
from scipy import linalg
from scipy.integrate import solve_ivp
import time
from time import perf_counter



def time_integration(Num):

    t_grid = np.linspace(start=0., stop=1., num=Num)

    def fun_rhs(t, y):
        dfdt = np.ones( len(t_grid) )
        return dfdt

    start_time = perf_counter()
    output = scipy.integrate.solve_ivp(fun_rhs, [ min(t_grid), max(t_grid) ], np.zeros( len(t_grid) ), method='BDF', t_eval=None, jac_sparsity=None, dense_output=False, events=None, vectorized=False, args=None, atol=1.e-2, rtol=1.e-2)
    stop_time = perf_counter()

    return stop_time - start_time


start_log = 1.
stop_log = 3.
timeSteps_array = np.logspace(start=start_log, stop=stop_log, num=int( stop_log-start_log ) + 1)

print(timeSteps_array)
print(int( stop_log-start_log ) + 1)


#timeSteps_array = np.logspace(start=1., stop=2., num=2)

start_time_script = perf_counter()
runtime_array = np.zeros( len(timeSteps_array) )
runtime_array = [time_integration( int(it) ) for it in timeSteps_array]
stop_time_script = perf_counter()


plt.plot(timeSteps_array, runtime_array)
plt.show()

print(f'the script took {stop_time_script - start_time_script} seconds to run.')
print(f'runtime = {runtime_array} sec')
