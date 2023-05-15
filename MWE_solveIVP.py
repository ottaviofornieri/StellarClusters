import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
import time
from time import perf_counter
from scipy.sparse import dia_array


dirName = '/Users/ottaviofornieri/PHYSICS_projects/GitProjects/StellarClusters/NumericalTests/'


def time_integration(Num):

    t_grid = np.linspace(start=0., stop=1., num=Num)
    data = [np.ones( Num, dtype=np.float64 )]
    jac_sparsity = dia_array( (data, [0]), shape=(Num, Num), dtype=np.float64 )

    def fun_rhs(t, y):
        dfdt = np.ones( len(t_grid) )
        return dfdt

    start_time = perf_counter()
    output = scipy.integrate.solve_ivp( fun_rhs, [ min(t_grid), max(t_grid) ], np.zeros( len(t_grid) ), 
                                       method='BDF', 
                                       t_eval=None, 
                                       jac_sparsity=jac_sparsity, 
                                       dense_output=False, 
                                       events=None,
                                       vectorized=False,
                                       args=None, atol=1.e-2, rtol=1.e-2 )
    stop_time = perf_counter()
    return stop_time - start_time, len(output.y)



start_log = 1.
stop_log = 7.
timeSteps_array = np.logspace(start=start_log, stop=stop_log, num=int( stop_log-start_log ) + 1, dtype=int)
print(f'array of time steps (len = {len(timeSteps_array)}): {timeSteps_array}')
print('')


start_time_script = perf_counter()
runtime_array = np.zeros( len(timeSteps_array) )
runtime_array = [time_integration( it )[0] for it in timeSteps_array]
stop_time_script = perf_counter()


print(f'runtime for each Nt = {runtime_array} sec')
print(f'length of the output = {[time_integration( it )[1] for it in timeSteps_array]}')
print(f'the script took {stop_time_script - start_time_script} seconds to run.')
plt.loglog(timeSteps_array, runtime_array, lw=2., color='blue', label='num test')
plt.loglog(timeSteps_array, time_integration( timeSteps_array[1] )[0] * timeSteps_array/timeSteps_array[1], lw=1.5, ls='--', color='red', label='linear')
plt.xlabel('Total number of points', fontsize=13)
plt.ylabel('Time [sec]', fontsize=13)
plt.legend(frameon=False, fontsize=12)
plt.savefig(dirName + 'TimeScaling.pdf',format='pdf',bbox_inches='tight', dpi=200)
plt.show()
