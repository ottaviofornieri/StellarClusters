import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rc, rcParams
f = mticker.ScalarFormatter(useMathText=True)
import scipy
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
import scipy.optimize as optimize
import scipy.integrate as integrate
from astropy.modeling import models
import time



###############
# LaTeX block #
###############
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Palatino']})
rc('xtick', labelsize=18)
rc('ytick', labelsize=18)
rcParams['legend.numpoints'] = 1



######################################
### Block to define some constants ###
######################################
conv_erg_GeV = 624.151
conv_pc_cm = 3.086e+18
conv_pc_m = 3.086e+16
conv_pc_km = 3.086e+13
conv_cm_pc = 3.24e-19
conv_yr_sec = 3.154e+7
conv_sec_yr = 3.17e-8
conv_mbarn_cm2 = 1.e-27         # Xsec conversion, millibarn to cm^2
conv_GeV_TeV = 1.e-3

m_p = 0.938272                  # proton mass, in [GeV c^{-2}]
m_p_grams = 1.67e-24            # proton mass, in [g]
m_e = 0.000510998918            # electron mass, in [GeV c^{-2}]
k_boltzmann = 8.617e-14         # Boltzmann constant, in [GeV K^(-1)]



## Set the precision when printing the numpy arrays ##
np.set_printoptions(precision=3)



def plot_cosmetics_single():
    ax = plt.gca()
    ax.tick_params(direction='in', axis='both', which='major', length=6.5, width=1.2, labelsize=18)
    ax.tick_params(direction='in', axis='both', which='minor', length=3., width=1.2, labelsize=18)
    ax.xaxis.set_tick_params(pad=7)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 15
    
def plot_cosmetics_multi():    
    ax = plt.gca()
    ax.tick_params(direction='in', axis='both', which='major', length=6.5, width=1.2, labelsize=20)
    ax.tick_params(direction='in', axis='both', which='minor', length=3., width=1.2, labelsize=20)
    ax.xaxis.set_tick_params(pad=10)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 10



## Function to compute the derivative of an array, even log spaced ##
def derivative_velocity(f_array_, var_array_):
    
    der_array = np.zeros( len(var_array_), dtype=np.float64 )
    for ir in range(1, len(var_array_) - 1):
        der_array[ir] = ( f_array_[ir+1] - f_array_[ir-1] ) / ( var_array_[ir+1] - var_array_[ir-1] )
        
    return der_array



###############################
## Parameters of the problem ##
###############################
n_ISM = 20.                               # density of the external ISM, in [cm^{-3}]
n_bubble = 1.                             # density between TS and FS, in [cm^{-3}]
M_dot = 1.5e-4                            # mass-loss rate, in [M_sun yr^{-1}]
v_0 = 2800.                               # wind velocity at the TS, in [km s^{-1}]
eta_B = 0.1                               # conversion effiency into turbulent B
L_c = 2.                                  # coherence length of the cluster field, in [pc]
delta_diff = 0.5
Z = 1                                     # atomic number, Z=1 for protons
p_max_cluster = 800.e3                    # 800 TeV, corresponding to a maximum 100 TeV photons, in [GeV]
t_physical_yr = 3.e+6                     # age of the system, in [yr]
t_run_yr = 3.e+6                           # run time, in [yr]

t_physical_Myr = t_physical_yr / 1.e6    # age of the system, in [Myr]
t_run_Myr = t_run_yr / 1.e6               # run time, in [Myr]


# magnetic field at the termination shock, in [muG]
B_TS = 3.7 * eta_B**(1/2) * (M_dot / 1.e-4)**(1/5) * (v_0 / 1.e3)**(2/5) * (n_ISM / 1.)**(3/10) * (t_physical_Myr / 10.)**(-2/5)

# wind luminosity, in [erg / s]
L_wind = 0.317e38 * (M_dot / 1.e-4) * (v_0 / 1.e3)**2
E_budget_CR = L_wind * t_physical_yr*conv_yr_sec   # energy budget into CRs, in [erg]
E_budget_CR_GeV = E_budget_CR * conv_erg_GeV

# reference length-scale, corresponding to the radius of the FS, in [pc]
L_ref = 175. * (L_wind / 1.e37)**(1/5) * (n_ISM / 1.)**(-1/5) * (t_physical_Myr / 10.)**(3/5)
def L_ref_func(t_):
    return 175. * (L_wind / 1.e37)**(1/5) * (n_ISM / 1.)**(-1/5) * (t_ / 10.)**(3/5)

# radius of the termination shock, in [pc]
R_TS = 48.6 * (M_dot / 1.e-4)**(3/10) * (v_0 / 1.e3)**(1/10) * (n_ISM / 1.)**(-3/10) * (t_physical_Myr / 10.)**(2/5)
def R_TS_func(t_):
    return 48.6 * (M_dot / 1.e-4)**(3/10) * (v_0 / 1.e3)**(1/10) * (n_ISM / 1.)**(-3/10) * (t_ / 10.)**(2/5)


# ram pressure of the wind, in [GeV / cm^3]
rho_wind_TS = 1.32e-27 * (M_dot / 1.e-4) * (R_TS / 20.)**(-2) * (v_0 / 1.e3)**(-1)       # in [g / cm^3]
ram_pressure_TS_erg = rho_wind_TS * (v_0 * 1.e5)**2                                      # in [erg / cm^3]
ram_pressure_TS_GeV = ram_pressure_TS_erg * conv_erg_GeV

# magnetic field according to Hillas criterion, in [muG]
B_Hillas = 0.642 * (p_max_cluster / 1.e5) * (1. / Z) * (20. / R_TS) * (2500. / v_0)

# diffusion coefficient immediately upstream, in [cm^2 / s]
if delta_diff == 0.:
    compr_diff_downstream = 1.                # D_upstream / D_downstream
    D_0 = 3.086e28 * (L_c / 1.)

elif delta_diff == 0.33:
    compr_diff_downstream = 0.67              # D_upstream / D_downstream
    D_0 = 5.73e27 * (L_c / 1.)**(2/3) * eta_B**(-1/6) * (M_dot / 1.e-4)**(-1/15) * (v_0 / 1.e3)**(-2/15) * (n_ISM / 1.)**(-1/10) * (t_physical_Myr / 10.)**(2/15) * (1 / Z)**(1/3)

elif delta_diff == 0.5:
    compr_diff_downstream = 0.55              # D_upstream / D_downstream
    D_0 = 2.47e27 * (L_c / 1.)**(1/2) * eta_B**(-1/4) * (M_dot / 1.e-4)**(-1/10) * (v_0 / 1.e3)**(-1/5) * (n_ISM / 1.)**(-3/20) * (t_physical_Myr / 10.)**(1/5) * (1 / Z)**(1/2)
    
elif delta_diff == 1.:
    compr_diff_downstream = 0.3               # D_upstream / D_downstream
    D_0 = 1.98e26 * eta_B**(-1/2) * (M_dot / 1.e-4)**(-1/5) * (v_0 / 1.e3)**(-2/5) * (n_ISM / 1.)**(-3/10) * (t_physical_Myr / 10.)**(2/5) * (1 / Z)
    
    

plt.figure(figsize=(5.5, 4.))
plot_cosmetics_single()

t_radius_Myr = np.logspace(start=-3., stop=1., num=100)
plt.loglog(t_radius_Myr, R_TS_func(t_radius_Myr), lw=2.5, color='blue', label='Termination Shock')
plt.loglog(t_radius_Myr, L_ref_func(t_radius_Myr), lw=2.5, color='red', label='Forward Shock')
plt.axvline(x=t_physical_Myr, ls='--', lw=1.5, color='orange', label='Run time')
plt.xlabel('$t \, [{\mathrm{Myr}}]$', fontsize=20)
plt.ylabel('$R \,  [{\mathrm{pc}}]$', fontsize=20)
plt.legend(fontsize=15, frameon=False, loc='best')
    

print(f'radius of the forward shock (FS), L_ref = {L_ref} [pc]')
print(f'radius of the termination shock (TS) = {R_TS} [pc]')
print(f'magnetic field at the TS = {B_TS} [muG]')
print(f'magnetic field according to Hillas criterion: B > {B_Hillas} [muG]')
print(f'diffusion coefficient at 10 TeV, D_0 = {D_0} [cm^2 / s]')
print(f'wind luminosity at the TS = {L_wind} [erg / s]')
print(f'wind density at the TS = {rho_wind_TS} [g / cm^3] = {rho_wind_TS / m_p_grams} protons / cm^3')
print('')
print(f'ram pressure of the wind = {ram_pressure_TS_GeV} [GeV / cm^3]')
print(f'energy budget injected by the source, available for CR acceleration = {E_budget_CR} [erg]')



#####################################################################
### Block for the assignments necessary to create the pot folders ###
#####################################################################
n_op = 2                 # number of variables in the PDE
L = 1.

# momentum grid
Np = 512
p_grid = np.logspace(start=2., stop=6., num=Np+1)       # in [GeV]
'''
Np = 4096
p_grid = np.logspace(start=2., stop=6., num=Np+1)       # in [GeV]
'''
deltap_over_p = [ (p_grid[ip+1] - p_grid[ip]) / p_grid[ip] for ip in range(len(p_grid)-1) ]
p_lower1 = p_grid[0] / ( 1 + deltap_over_p[0] )           # ghost momentum, lower
p_higher1 = p_grid[-1] * ( 1 + deltap_over_p[0] )         # ghost momentum, upper
dp_min = np.min([p_grid[ip+1] - p_grid[ip] for ip in range(len(p_grid)-1)])
p_ref = 1.



###########################################################
### Create the folders to store all the plots and files ###
###########################################################
path_PDE = '/Users/ottaviofornieri/PHYSICS_projects/GitProjects/StellarClusters/'
solutions_folder_name = 'Radau_SphericalShock_Tage=' + str("{:.1f}".format(t_physical_Myr)) + 'Myr_Trun=' + str("{:.1f}".format(t_run_Myr)) + 'Myr_nISM=' + str(n_ISM) + '_nBubble=' + str(n_bubble) + '_Np=' + str(Np) + '_deltaDiff=' + str(delta_diff)
dirName = path_PDE + 'Stored_solutions/' + str(solutions_folder_name) + '/'

try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory", dirName, "created")
    print("")
except FileExistsError:
    print("Directory", dirName, "already exists")
    print("")



#########################################
### Block for preliminary assignments ###
#########################################

# Building the diffusion coefficient: D(p) = D_0*(p/p_0)^delta #
################################################################
p_0_diff = 1.e+4                 # in [GeV]

def D_coeff_func(p_):
    return D_0 * (p_ / p_0_diff)**delta_diff

D_ref = np.min(D_coeff_func(p_grid))
pdot_ref = p_ref / ( (L_ref*conv_pc_cm)**2 / D_ref )
################################################################


# velocity of the wind, for advection
#####################################
v_0_cm = v_0*1.e5                                 # in [cm s^{-1}]
v_dless = v_0_cm * (L_ref*conv_pc_cm) / D_ref     # dimensionless velocity upstream
compr_factor = 4.                                 # compression factor at the shock
DeltaL_v = 0.1 * L_ref                            # step for the change in the velocity profile, in [pc]
DeltaL_v_dless = DeltaL_v / L_ref
#####################################


DeltaX_acc_up = D_coeff_func(p_grid) / v_0_cm * conv_cm_pc                      # acceleration length upstream
DeltaX_acc_down = D_coeff_func(p_grid) / (v_0_cm/compr_factor) * conv_cm_pc     # acceleration length downstream
factor_below_DeltaXAcc_maxDx = 1.
factor_below_DeltaXAcc_minDx = 1. #10.
DeltaX_acc_min = min( min(DeltaX_acc_up), min(DeltaX_acc_down) )
DeltaX_acc_max = max(D_coeff_func(p_grid)) / v_0_cm * conv_cm_pc
DeltaX_acc_min_dless = DeltaX_acc_min / L_ref
DeltaX_acc_max_dless = DeltaX_acc_max / L_ref
necessary_dx = DeltaX_acc_min_dless / factor_below_DeltaXAcc_minDx
necessary_Nx = int( L/necessary_dx )
width_at_shock = 2         # region with the finest coarse, around the shock, width_at_shock/2 + width_at_shock/2, in dimensionless units D/u
dx_min = necessary_dx
dx_max = DeltaX_acc_min_dless / factor_below_DeltaXAcc_maxDx
dx_max_down = (min(DeltaX_acc_down) / L_ref) / factor_below_DeltaXAcc_maxDx
dx_physical_min = dx_min * L_ref                          # minimum spatial step, in [pc]
dx_physical_max = dx_max * L_ref                          # maximum spatial step, in [pc]
dx_physical_max_down = dx_max_down * L_ref                # maximum spatial step downstream, in [pc]


num_sections = L_ref / R_TS                               # number of sections to slice the spatial grid
start_x_grid_slices = 0.                                  # starting point of the space grid
shock_location = start_x_grid_slices + L/num_sections     # location of the shock, in terms of slices of the grid
left_space = shock_location - DeltaX_acc_max_dless * (width_at_shock/2)
right_space = L - (shock_location + DeltaX_acc_max_dless * (width_at_shock/2))



if left_space >= 0.:
    Nx_around_shock_left = round(DeltaX_acc_max_dless * (width_at_shock/2) / dx_min)
    Nx_around_shock_right = Nx_around_shock_left
    Nx_away_from_shock_left = round(left_space / dx_max)
else:
    Nx_around_shock_left = round((shock_location - start_x_grid_slices) / dx_min)
    Nx_around_shock_right = round(DeltaX_acc_max_dless * (width_at_shock/2) / dx_min)
    Nx_away_from_shock_left = 0
    
if Nx_away_from_shock_left == 0: Nx_away_from_shock_left = 2
Nx_away_from_shock_right = round(right_space / dx_max_down)
if Nx_away_from_shock_right < 5: Nx_away_from_shock_right = 5


indx_intersec_tauAcc_upstream = np.argmin( abs(DeltaX_acc_up - shock_location*L_ref) )
    

print(f'maximum momentum allowed by the upstream region = {p_grid[indx_intersec_tauAcc_upstream]} [GeV]')
print(f'number of sections in which we divided the space: {num_sections}')
print(f'acceleration length: {DeltaX_acc_max_dless * (width_at_shock/2)}, left space: {left_space}')
print(f'dimensionless location of the shock: {shock_location}')
print(f'DeltaX_acc minimum = {DeltaX_acc_min} [pc], DeltaX_acc maximum = {DeltaX_acc_max} [pc]')
print(f'dimensionless DeltaX_acc minimum = {DeltaX_acc_min_dless}, dimensionless DeltaX_acc maximum = {DeltaX_acc_max_dless}')
print(f'factors below the minimum acceleration scale = {factor_below_DeltaXAcc_minDx}, {factor_below_DeltaXAcc_maxDx}')
print(f'necessary dimensionless dx = {necessary_dx}, necessary Nx with a uniform grid = {necessary_Nx}')
print(f'Nx around the shock (left) = {Nx_around_shock_left}, Nx around the shock (right) = {Nx_around_shock_right}, Nx away from the shock (left) = {Nx_away_from_shock_left}, Nx away from the shock (right) = {Nx_away_from_shock_right}, total Nx = {Nx_around_shock_left + Nx_around_shock_right + Nx_away_from_shock_left + Nx_away_from_shock_right}')
print(f'min physical space-step: {dx_physical_min} [pc]')
print(f'max physical space-step: {dx_physical_max} [pc]')
print('')



plt.figure(figsize=(10.5, 4.5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.title('Upstream', fontsize=18, pad=8)
plt.loglog(p_grid, DeltaX_acc_up, lw=2.5, color='blue', label='$\\delta = \;$' + str(delta_diff)) 
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$\\frac{D(p)}{u} \, [\mathrm{pc}]$',fontsize=20)
plt.axhline(y=shock_location*L_ref, ls='--', lw=1.5, color='red', label='$\\Delta r_{\mathrm{up}} = \,$' + str("{:.1f}".format((shock_location)*L_ref)) + '\, pc')
plt.axhline(y=dx_physical_min, ls='--', lw=1.5, color='orange', label='$dr_{\mathrm{min}} = \;$' + str("{:.3f}".format(dx_physical_min)) + '\, pc')
plt.axhline(y=dx_physical_max, ls='--', lw=1.5, color='green', label='$dr_{\mathrm{max, up}} = \;$' + str("{:.3f}".format(dx_physical_max)) + '\, pc')
plt.axvline(x=p_grid[indx_intersec_tauAcc_upstream], ls='--', color='magenta', label='$p_{\mathrm{max, up}} \simeq \,$' + str("{:.0f}".format(p_grid[indx_intersec_tauAcc_upstream]/1.e3)) + '\, TeV')
plt.legend(fontsize=16, frameon=False)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

plt.title('Downstream', fontsize=18, pad=8)
plt.loglog(p_grid, DeltaX_acc_down, lw=2.5, color='blue', label='$\\delta = \;$' + str(delta_diff))
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$\\frac{D(p)}{u} \, [\mathrm{pc}]$',fontsize=20)
plt.axhline(y=(L-shock_location)*L_ref, ls='--', lw=1.5, color='red', label='$\\Delta r_{\mathrm{down}} = \,$' + str("{:.1f}".format((L-shock_location)*L_ref)) + '\, pc')
plt.axhline(y=dx_physical_min, ls='--', lw=2., color='orange', label='$dr_{\mathrm{min}} = \;$' + str("{:.3f}".format(dx_physical_min)) + '\, pc')
plt.axhline(y=dx_physical_max_down, ls='--', lw=1.5, color='green', label='$dr_{\mathrm{max, down}} = \;$' + str("{:.3f}".format(dx_physical_max_down)) + '\, pc')
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()
plt.savefig(dirName + 'AccelerationLength.pdf',format='pdf',bbox_inches='tight', dpi=200)



# dimensionless version
plt.figure(figsize=(10.5, 4.5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.title('Upstream, dimensionless', fontsize=18, pad=8)
plt.loglog(p_grid, DeltaX_acc_up / L_ref, lw=2.5, color='blue', label='$\\delta = \;$' + str(delta_diff))
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$\\frac{D(p)}{u}$',fontsize=20)
plt.axhline(y=dx_min, ls='--', lw=2., color='orange', label='$dr_{\mathrm{min}} = \;$' + str("{:.5f}".format(dx_min)))
plt.legend(fontsize=16, frameon=False)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

plt.title('Downstream, dimensionless', fontsize=18, pad=8)
plt.loglog(p_grid, DeltaX_acc_down / L_ref, lw=2.5, color='blue', label='$\\delta = \;$' + str(delta_diff))
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$\\frac{D(p)}{u}$',fontsize=20)
plt.axhline(y=dx_min, ls='--', lw=2., color='orange', label='$dr_{\mathrm{min}} = \;$' + str("{:.5f}".format(dx_min)))
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()



# non-uniform space grid #
##########################
start_x_grid = start_x_grid_slices
x_around_shock = DeltaX_acc_max_dless * (width_at_shock/2)
stop_x_grid = start_x_grid + L

if left_space >= 0.:
    x_grid = np.r_[np.linspace(start_x_grid, left_space, Nx_away_from_shock_left, endpoint=False), 
               np.linspace(left_space, shock_location + DeltaX_acc_max_dless * (width_at_shock/2), Nx_around_shock_left + Nx_around_shock_right, endpoint=False),
               np.linspace(shock_location + DeltaX_acc_max_dless * (width_at_shock/2), stop_x_grid, Nx_away_from_shock_right, endpoint=True)]
else:
    x_grid = np.r_[np.linspace(start_x_grid, shock_location, Nx_around_shock_left, endpoint=False), 
               np.linspace(shock_location, shock_location + DeltaX_acc_max_dless * (width_at_shock/2), Nx_around_shock_right, endpoint=False),
               np.linspace(shock_location + DeltaX_acc_max_dless * (width_at_shock/2), stop_x_grid, Nx_away_from_shock_right, endpoint=True)]
    
    

x_grid = np.asarray(x_grid)
dx_array = [(x_grid[ix+1] - x_grid[ix]) for ix in range(0, len(x_grid)-1)]
Nx = len(x_grid) - 1
dx_min_nu = min(dx_array)
dx_low = x_grid[1] - x_grid[0]
dx_up = x_grid[-1] - x_grid[-2]
shock_index = np.argmin( abs(shock_location - x_grid) )


print(f'length of the non-uniform grid (Nx) = {len(x_grid)}')
print(f'first grid point = {x_grid[0]}, last grid point = {x_grid[-1]}')
print(f'shock_index: {shock_index}, x_grid[shock_index] = {x_grid[shock_index]} = {x_grid[shock_index]*L_ref} [pc]')
print(f'minimum dx = {dx_min}, maximum dx = {dx_max}')
print(f'minimum dx in the non-uniform grid = {dx_min_nu}')
print(f'length of the momentum grid (Np) = {len(p_grid)}')
print(f'length of the linearized 2D grid (Nx * Np) = {len(x_grid) * len(p_grid)}')
print('')


plt.figure(figsize=(5.5, 4.))
plot_cosmetics_single()


plt.title('Non-uniform grid', fontsize=18, pad=8)
plt.scatter(x_grid, [1. for i in x_grid], s=5, color='blue')
plt.axvline(x=shock_location, ls='--', lw=1.5, color='blue', label='shock')
plt.axvline(x=shock_location-width_at_shock/2 * DeltaX_acc_max_dless, ls='--', lw=1.5, color='Orange', label='$\mathrm{avg} - \\frac{D_{\mathrm{max}}}{u_{\mathrm{up}}}$')
plt.axvline(x=shock_location+width_at_shock/2 * DeltaX_acc_max_dless, ls='--', lw=1.5, color='magenta', label='$\mathrm{avg} + \\frac{D_{\mathrm{max}}}{u_{\mathrm{up}}}$')
plt.xlabel('$r \in [0, \, R_{\mathrm{forward}}]$', fontsize=20)
plt.yticks([])
plt.legend(fontsize=15, frameon=False, loc='upper right')
plt.text(0.6, 0.1, '$\\frac{D_{\mathrm{max}}}{u_{\mathrm{up}}}\\big|_{\mathrm{dless}} = \;$' + str("{:.3f}".format(DeltaX_acc_max_dless)), fontsize=15, transform = plt.gca().transAxes)
plt.text(0.6, 0.25, '$\\frac{D_{\mathrm{max}}}{u_{\mathrm{up}}} = \;$' + str("{:.3f}".format(DeltaX_acc_max)) + '$\, \mathrm{pc}$', fontsize=15, transform = plt.gca().transAxes)
plt.text(shock_location - 0.6*shock_location, 1.003, '$N_r^{\mathrm{shock}} = \,$' + str(Nx_around_shock_left+Nx_around_shock_right), fontsize=12, rotation=90)
plt.text(shock_location + 1.1*width_at_shock/2 * DeltaX_acc_max_dless, 1.003, '$N_r^{\mathrm{right}} = \,$' + str(Nx_away_from_shock_right), fontsize=12, rotation=0)
if left_space > 0.:
    plt.text(shock_location - 1.1*width_at_shock/2 * DeltaX_acc_max_dless, 1.003, '$N_r^{\mathrm{left}} = \,$' + str(Nx_away_from_shock_left), fontsize=12, rotation=90)

plt.savefig(dirName + 'NonUniform_Grid.pdf',format='pdf',bbox_inches='tight', dpi=200)
##########################

# creating the D(E) at each point in space
D_matrix = np.zeros( (len(x_grid), len(p_grid)), dtype=np.float64 )
D_matrix_dless = np.zeros( (len(x_grid), len(p_grid)), dtype=np.float64 )


for ix in range(len(x_grid)):
    if ix <= shock_index:
        D_matrix[ix, :] = D_coeff_func(p_grid[:]) * ( R_TS / ( R_TS + abs(R_TS - x_grid[ix]*L_ref) ) )**(delta_diff)
    else:
        D_matrix[ix, :] = D_coeff_func(p_grid[:]) * compr_diff_downstream
        
    D_matrix_dless[ix, :] = D_matrix[ix, :] / D_ref

    
# linear extrapolation of the diffusion-coefficient matrix one point above and below the space grid
D_SpaceSlope_up = (D_matrix[-1, 0] - D_matrix[-2, 0]) / dx_up
D_SpaceSlope_low = (D_matrix[1, 0] - D_matrix[0, 0]) / dx_low
D_higher1 = [D_matrix[-1, 0] + D_SpaceSlope_up * dx_up for ip in range(len(p_grid))]
D_lower1 = [D_matrix[0, 0] - D_SpaceSlope_low * dx_low for ip in range(len(p_grid))]



## velocity profiles, for advection ##
######################################
#case_velocity = 'spherical_velocity_SolarWind'
case_velocity = 'spherical_velocity'

def v_profile_func(x_):
    
    v_profile = np.zeros( len(x_), dtype=np.float64 )
    
    if case_velocity == 'symmetric_velocity':
        center = x_[shock_index]
        dist_from_center = DeltaL_v_dless
        indx_temp = [ix for ix in range(len(x_)) if (abs(x_[ix] - center) <= dist_from_center)]

        for ix in range(len(x_)):
            if (abs(x_[ix] - center) > dist_from_center):
                v_profile[ix] = v_dless * (1./( abs(x_[max(indx_temp)+1] - center) )**(-2)) * ( abs(x_[ix] - center) )**(-2)
            else:
                v_profile[ix] = v_dless
                
        label_v_profile = '$v(x) = v_{0} \cdot \\left( \\frac{x - x_{\mathrm{inj}}}{\\Delta x_{v}} \\right)^{-2}$'
    
        
    elif case_velocity == 'spherical_velocity':
        shock_loc = x_[shock_index]
        
        for ix in range(0, len(x_)):
            if ix < shock_index:
                v_profile[ix] = v_dless
            else:
                v_profile[ix] = v_dless/compr_factor * ( x_[ix] / shock_loc )**(-2)
                
        label_v_profile = 'Spherical_Cluster'
        
        
    elif case_velocity == 'spherical_velocity_modified':
        shock_loc = x_[shock_index]
        
        for ix in range(1, len(x_)):
            if ix < shock_index:
                v_profile[ix] = v_dless * ( x_[ix] / shock_loc )**(2)
            else:
                v_profile[ix] = v_dless/compr_factor * ( x_[ix] / shock_loc )**(-2)
                
        v_profile[0] = v_profile[1]
        label_v_profile = 'Spherical_Cluster_Modified'
        
        
    elif case_velocity == 'spherical_velocity_SolarWind':
        shock_loc = x_[shock_index]
        v_dless_initial = v_dless/4
        m_slope = (v_dless - v_dless_initial) / (shock_loc - 0)
        
        v_profile[0] = v_dless_initial
        
        for ix in range(1, len(x_)):
            if ix < shock_index:
                v_profile[ix] = v_profile[ix-1] + m_slope * ( x_[ix] -  x_[ix-1] )
            else:
                v_profile[ix] = v_dless/compr_factor * ( x_[ix] / shock_loc )**(-2)
                
        label_v_profile = 'Spherical_Cluster_SolarWind'
        
        
    elif case_velocity == 'step_function':
        shock_loc = x_[shock_index]
        v_profile = [v_dless / compr_factor if x_[ix] >= shock_loc else v_dless for ix in range(len(x_))]
        label_v_profile = 'Step function'
        
        
    elif case_velocity == 'tanh_step_function':
        ## to be corrected: the tanh is centered at the central point of the grid ##
        center = x_[shock_index]
        v_profile = [v_dless/2 * (1 + np.tanh(-x_[ix] / (dx_min))) + v_dless/(2*compr_factor) * (1 + np.tanh(x_[ix] / (dx_min))) for ix in range(len(x_))]
        label_v_profile = 'tanh'
        
        
    elif case_velocity == 'Galactic':
        center = x_[len(x_)//2]
        v_profile = [v_dless * np.tanh(x_[ix] / (dx_min)) for ix in range(len(x_))]
        label_v_profile = 'Galactic'
        
            
    elif case_velocity == 'constant_velocity':
        v_profile = [v_dless for ix in range(len(x_))]
        label_v_profile = '$v(x) = v_0$'
        
        
    elif case_velocity == 'linear_growth_velocity':
        v_profile = [v_dless + v_dless * ( abs( x_[ix] - min(x_) ) / DeltaL_v_dless ) for ix in range(len(x_))]
        label_v_profile = '$v(x) = v_0 \cdot \\left( \\frac{x}{\\Delta x_{v}} \\right)$'
        
        
    elif case_velocity == 'quadratic_growth_velocity':
        v_profile = [v_dless + v_dless * ( abs( x_[ix] - min(x_) ) / DeltaL_v_dless )**(2.) for ix in range(len(x_))]
        label_v_profile = '$v(x) = v_0 \cdot \\left( \\frac{x}{\\Delta x_{v}} \\right)^2$'
            
            
    return v_profile, label_v_profile



label_v_profile = v_profile_func(x_grid)[1]
v_adv = v_profile_func(x_grid)[0]
v_adv_Rsquare = v_adv * x_grid**2
v_adv_derivative = derivative_velocity(v_profile_func(x_grid)[0], x_grid)
v_adv_derivative_Rsquare = derivative_velocity(v_adv_Rsquare, x_grid)
######################################



## profile of the densities ##
##############################

# wind density
def wind_density(r_):
    return (rho_wind_TS / m_p_grams) * ( r_ / (R_TS/L_ref) )**(-2)   # in [particles / cm^3]


wind_density_array = np.zeros( len(x_grid), dtype=np.float64 )
for ix in range(len(x_grid)):
    if x_grid[ix] <= (L_c/L_ref):
        wind_density_array[ix] = wind_density(x_grid[ np.argmin( abs(x_grid - L_c/L_ref) ) ])
    else:
        wind_density_array[ix] = wind_density(x_grid[ix])
        
        
# density of the ISM
n_bubble_array = np.zeros( len(x_grid), dtype=np.float64 )
for ix in range(len(x_grid)):
    if ix < shock_index:
        n_bubble_array[ix] = n_bubble / compr_factor
    else:
        n_bubble_array[ix] = n_bubble
##############################



#############################################################################
### Block to define the rate of momentum loss, computed in CGS base units ###
#############################################################################
# Klein-Nishina factor from Evoli et al. arXiv:2007.01302 (2020) #


ISRF_components = ['CMB', 'IR', 'OPT', 'UV_1', 'UV_2', 'UV_3']
T_ISRF = [2.725, 33.07, 313.32, 3249.3, 6150.4, 23209.0]            # in [K]
U_ISRF = [0.26e-9, 0.25e-9, 0.055e-9, 0.37e-9, 0.23e-9, 0.12e-9]    # energy density of the contributions, in [GeV cm^(-3)]

sigma_Thomson = 6.65e-25                             # in [cm^2]
c_cm = 2.99e+10                                      # speed of light, in [cm s^(-1)]
factor_ratio = ( 45 / (64*np.pi**2) )


# Leptons
def loss_rate_KN(momentum_, B_):
# momentum variable in [GeV], B field in [muG], => result in [GeV * s^{-1}]
    
    f_KN_times_U = 0.
    for i in range (len(T_ISRF)):

        f_KN_times_U_single = ( (factor_ratio * ( m_e / (k_boltzmann*T_ISRF[i]) )**2) / ( (factor_ratio * ( m_e / (k_boltzmann*T_ISRF[i]) )**2) + (momentum_/m_e)**2 ) ) * U_ISRF[i]
        #f_KN_times_U_single = U_ISRF[i]     # uncomment for the Thomson limit
        f_KN_times_U = f_KN_times_U + f_KN_times_U_single
        
    U_B = ( (B_*1.e-6)**2 / (8*np.pi) ) * conv_erg_GeV
    dp_dt_KN = - (4/3) * (sigma_Thomson * c_cm) * (f_KN_times_U + U_B) * (momentum_/m_e)**2
    return dp_dt_KN



## define the spatial-dependent magnetic field in the region
def B_field(x_):
    # result in [muG]
    return ( x_ / shock_location )**(-1.) * B_TS

B_space = np.zeros( len(x_grid), dtype=np.float64 )
for ix in range(len(x_grid)):
    if x_grid[ix] <= (L_c/L_ref):
        B_space[ix] = B_field(x_grid[ np.argmin( abs(x_grid - L_c/L_ref) ) ])
    elif x_grid[ix] > (L_c/L_ref) and x_grid[ix] <= shock_location:
        B_space[ix] = B_field(x_grid[ix])
    else:
        B_space[ix] = B_field(shock_location) / np.sqrt(11)

B_field_indx_max = np.argmax( B_space )


# momentum loss-rate in each location
loss_rate_KN_matrix = np.zeros( (len(x_grid), len(p_grid)), dtype=np.float64 )
for ix in range(len(x_grid)):
    for ip in range(len(p_grid)):
        
        loss_rate_KN_matrix[ix, ip] = loss_rate_KN(p_grid[ip], B_space[ix])

        
        
# Hadrons
def loss_rate_pp_collisions(indx_r_, momentum_):
# momentum variable in [GeV], result in [GeV * s^{-1}]

    L_log = np.log(momentum_ / 1.e3)
    K_pi = 0.13
    return -5.1e-15 * K_pi * n_bubble_array[indx_r_] * (momentum_ / 1.) * (1 + 5.5e-2 * L_log + 7.3e-3 * L_log**2)
    
loss_rate_pp_collision_matrix = np.zeros( (len(x_grid), len(p_grid)), dtype=np.float64 )
for ix in range(len(x_grid)):
    loss_rate_pp_collision_matrix[ix, :] = [loss_rate_pp_collisions(ix, p_grid[ip]) for ip in range(len(p_grid))]


    

plt.figure(figsize=(13, 4.5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()


plt.loglog(p_grid, np.min(abs(loss_rate_KN(p_grid, B_space[shock_index]))) * (p_grid / min(p_grid))**2, lw=2.5, color='blue', label='$\propto p^2$ loss rate')
plt.loglog(p_grid, abs(loss_rate_KN(p_grid, B_space[shock_index])), lw=2.5, color='red', label='$\\approx$ Klein-Nishina')
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$ |\dot{p}| \, [\mathrm{GeV} \cdot \mathrm{s}^{-1}]$',fontsize=20)
plt.text(0.65, 0.1, '$B = \,$' + str("{:.1f}".format(B_space[shock_index])) + '$\, \mu \mathrm{G}$', fontsize=18, transform = plt.gca().transAxes)
plt.legend(fontsize=18, frameon=False)



# same but with dimensionless quantities
plt.subplot(1, 2, 2)
plot_cosmetics_multi()


plt.loglog(p_grid, np.min(abs((loss_rate_KN(p_grid, B_space[shock_index]) / pdot_ref))) * (p_grid / min(p_grid))**2, lw=2.5, color='blue', label='$\propto p^2$ loss rate')
plt.loglog(p_grid, abs((loss_rate_KN(p_grid, B_space[shock_index]) / pdot_ref)), lw=2.5, color='red', label='$\\approx$ Klein-Nishina')
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$ |\dot{p}| \\big/ \dot{p}_{\mathrm{ref}}$',fontsize=20)
plt.legend(fontsize=18, frameon=False)
plt.text(0.65, 0.25, '$B = \,$' + str("{:.1f}".format(B_space[shock_index])) + '$\, \mu \mathrm{G}$', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.35, 0.1, '$\\dot{p}_{\mathrm{ref}} = \,$' + str('{:.2e}'.format(pdot_ref)) + ' $\mathrm{GeV} \cdot \mathrm{s}^{-1}$ ', fontsize=18, transform = plt.gca().transAxes)
plt.tight_layout()



# compute the diffusive distance after energy losses #
# leptons
def tau_func_KN(p_, B_):
    return - 1. / abs(loss_rate_KN(p_, B_))

def tau_func_Th(p_, B_):
    return - 1. / ( np.min(abs(loss_rate_KN(p_, B_))) * (p_ / min(p_grid))**2 )

# hadrons
def tau_pp_coll(indx_r_, momentum_):
    return - 1. / abs( loss_rate_pp_collisions(indx_r_, momentum_) )




## example, for a given magnetic field ##
momentum_ref_loss = 1.e+2      # in [GeV]
momentum_ref_loss_ind = np.argmin(abs(p_grid - momentum_ref_loss))
space_point_diff = 0

B_field = B_space[shock_index]         # magnetic field, in [muG]
U_B = ( (B_field*1.e-6)**2 / (8*np.pi) ) * conv_erg_GeV

list_integral_tau_KN = np.logspace(start=np.log10(momentum_ref_loss*10000), stop=np.log10(momentum_ref_loss), num=1000)
integral_tau_KN = np.trapz(tau_func_KN(list_integral_tau_KN, B_space[shock_index]), list_integral_tau_KN, axis=-1)   # in [sec]
diff_distance = np.sqrt( 4 * D_matrix[space_point_diff, momentum_ref_loss_ind] * integral_tau_KN ) * conv_cm_pc


print('** Leptonic losses: Example **')
print('')
print('B field =', B_field, '[muG], magnetic energy density =', U_B, '[GeV cm^{-3}]')
print('')
print('E*tau at', momentum_ref_loss, '[GeV/c]:', momentum_ref_loss * integral_tau_KN*conv_sec_yr/1.e6, '[GeV/c Myr]')
print('Loss timescale =', integral_tau_KN, '[s] =', integral_tau_KN*conv_sec_yr/1.e6, '[Myr]')
print('Diffusive distance for ' + str(momentum_ref_loss) + ' [GeV] leptons =', diff_distance, '[pc]')
print('')
#########################################


integrals_losses_KN = np.zeros( len(p_grid), dtype=np.float64 )
integrals_losses_Th = np.zeros( len(p_grid), dtype=np.float64 )
integrals_losses_ppcoll = np.zeros( (len(x_grid), len(p_grid)), dtype=np.float64 )
for ip in range(len(p_grid)):
    # leptons
    list_integral_tau_KN_temp = np.logspace(start=np.log10(p_grid[ip]*10000), stop=np.log10(p_grid[ip]), num=1000)
    integrals_losses_KN[ip] = np.trapz(tau_func_KN(list_integral_tau_KN_temp, B_space[shock_index]), list_integral_tau_KN_temp, axis=-1)
    
    # hadrons
    list_integral_tau_ppcoll = np.logspace(start=np.log10(p_grid[ip]*10000), stop=np.log10(p_grid[ip]), num=1000)
    integrals_losses_ppcoll[:, ip] = [np.trapz(tau_pp_coll( ix, list_integral_tau_ppcoll ), list_integral_tau_ppcoll, axis=-1) for ix in range(len(x_grid))]

    

# definition of the timescales #
# leptons
timescale_losses = integrals_losses_KN*conv_sec_yr/1.e+6
timescale_diff = (L_ref*conv_pc_cm)**2 / (2.* D_matrix[shock_index, :])*conv_sec_yr/1.e+6
timescale_adv = [1./3. * (L_ref*conv_pc_cm) / (v_0_cm / compr_factor) * (L_ref / R_TS) * (1 - (R_TS / L_ref)**3.)*conv_sec_yr/1.e+6 for i in range(len(p_grid))]
timescale_acc_sec = (3. / (v_0_cm - v_0_cm/compr_factor) * ( (D_matrix[0, :] / v_0_cm) + (D_matrix[0, :] / (v_0_cm/compr_factor)) ))
timescale_acc = timescale_acc_sec * conv_sec_yr/1.e6
# hadrons
timescale_pp_losses = integrals_losses_ppcoll*conv_sec_yr/1.e+6

# leptons
indx_intersec_LossDiff = np.argmin( abs(timescale_losses - timescale_diff) )
indx_intersec_LossAdv = np.argmin( abs(timescale_losses - timescale_adv) )
indx_intersec_LossAcc = np.argmin( abs(timescale_losses - timescale_acc) )
# hadrons
indx_intersec_ppLossAcc = np.zeros( len(x_grid), dtype=np.float64 )
indx_intersec_ppLossAcc = [np.argmin( abs(timescale_pp_losses[ix] - timescale_acc) ) for ix in range(len(x_grid))]
##########################



plt.figure(figsize=(13, 5.))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.title('Leptons', fontsize=18, pad=8)
plt.loglog(p_grid, timescale_losses, lw=2., color='red', label='$\\approx$ Klein-Nishina, $B = \,$' + str("{:.1f}".format(B_space[shock_index])) + '$\, \mu \mathrm{G}$')
plt.loglog(p_grid, timescale_diff, lw=2., color='blue', label='$\\tau_{\mathrm{diff}} = \\frac{L^2_{\mathrm{ref}}}{2 \cdot D(p)}, \; \\delta = \,$' + str(delta_diff))
plt.loglog(p_grid, timescale_adv, lw=2., color='green', label='$\\tau_{\mathrm{adv}} = \\int^{R_{\mathrm{b}}}_{R_{\mathrm{TS}}} \\frac{dr}{v_{\mathrm{d}}(r)}, \; v_{\mathrm{d}} = \,$'+ str("${}$".format(f.format_data(v_0/compr_factor))) + '$\, \mathrm{km \cdot s^{-1}}$')
plt.loglog(p_grid, timescale_acc, lw=2., color='orange', label='$\\tau_{\mathrm{acc}}$')
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$\\tau \, [\mathrm{Myr}]$',fontsize=20)
plt.legend(fontsize=15, frameon=False)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

pp_coll_loss_indx = shock_index
plt.title('Protons', fontsize=18, pad=8)
plt.loglog(p_grid, timescale_pp_losses[pp_coll_loss_indx,:], lw=2., color='red', label='$\\tau_{pp}, \, n_{\mathrm{bubble}} = \,$' + str("{:.1f}".format(n_bubble_array[pp_coll_loss_indx])) + '$\, \mathrm{cm^{-3}}$')
plt.loglog(p_grid, timescale_diff, lw=2., color='blue', label='$\\tau_{\mathrm{diff}} = \\frac{L^2_{\mathrm{ref}}}{2 \cdot D(p)}, \; \\delta = \,$' + str(delta_diff))
plt.loglog(p_grid, timescale_adv, lw=2., color='green', label='$\\tau_{\mathrm{adv}} = \\int^{R_{\mathrm{b}}}_{R_{\mathrm{TS}}} \\frac{dr}{v_{\mathrm{d}}(r)}, \; v_{\mathrm{d}} = \,$'+ str("${}$".format(f.format_data(v_0/compr_factor))) + '$\, \mathrm{km \cdot s^{-1}}$')
plt.loglog(p_grid, timescale_acc, lw=2., color='orange', label='$\\tau_{\mathrm{acc}}$')
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$\\tau \, [\mathrm{Myr}]$',fontsize=20)
plt.legend(fontsize=15, frameon=False)

plt.tight_layout()
plt.savefig(dirName + 'Timescales.pdf',format='pdf',bbox_inches='tight', dpi=200)



#########################
## Settings of the run ##
#########################
# time grid
Tbar_dim_s = (L_ref*conv_pc_cm)**2 / D_ref               # time unit in [sec], to make time dimensionless


print(f'Nx = {Nx}, Np = {Np}')
print(f'Nk = Nx * Np = {len(x_grid) * len(p_grid)}')
print('')
print('** Space information **')
print(f'min dimensionless dx = {dx_min}')
print(f'min physical space step = {dx_physical_min} [pc]')
print(f'max dimensionless dx = {dx_max}')
print(f'max physical space step = {dx_physical_max} [pc]')
print('')
print('** Time information **')
print(f'minimum diffusive timescale: {min(timescale_diff)} [Myr]')
print(f'minimum advective timescale: {min(timescale_adv)} [Myr]')
print(f'minimum losses timescale: {min(timescale_losses)} [Myr]')
print(f'age of the system, T_age = {t_physical_Myr} [Myr]')
print(f'duration of the run, T_tot = {t_run_Myr} [Myr]')
print('')
print('** Momentum information **')
print(f'min momentum = {p_grid[0]} GeV, max momentum = {p_grid[-1]} GeV')
print(f'minimum step in the momentum grid = {dp_min} [GeV]')
print('')
print('** Acceleration information **')
print(f'minimum/maximum acceleration timescale: {min(timescale_acc)} [Myr] / {min(timescale_acc)} [Myr]')



plt.figure(figsize=(14, 5))

# diffusion coefficient subplots
plt.subplot(1, 2, 1)
plot_cosmetics_multi()


loc_diff_coeff_upstream = np.linspace(start=start_x_grid, stop=shock_location, num=5, endpoint=False)
loc_diff_coeff_downstream = np.linspace(start=shock_location, stop=stop_x_grid, num=3)
loc_diff_coeff = np.concatenate((loc_diff_coeff_upstream, loc_diff_coeff_downstream))
indx_diff_coeff = [np.argmin( abs(loc_diff_coeff[ix] - x_grid) ) for ix in range(len(loc_diff_coeff))]

for ix in range(len(loc_diff_coeff)):
    plt.loglog(p_grid, D_matrix[indx_diff_coeff[ix], :], lw=2.5, label='$r = \, $' + str("{:.2f}".format(L_ref*x_grid[indx_diff_coeff[ix]])) + '$\, \mathrm{pc}$')

plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$ D(p) \, [\mathrm{cm^2} \cdot \mathrm{s^{-1}}]$',fontsize=20)
plt.text(0.55, 0.1, '$r_{\mathrm{shock}} = \,$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=19, transform = plt.gca().transAxes)
plt.legend(fontsize=18, frameon=False, loc='upper left')


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

en_diff_coeff = np.logspace(start=np.log10(min(p_grid)), stop=np.log10(max(p_grid)), num=3)
indx_en_diff_coeff = [np.argmin( abs(en_diff_coeff[ip] - p_grid) ) for ip in range(len(en_diff_coeff))]


for ip in range(len(en_diff_coeff)):
    plt.plot(x_grid, D_matrix[:, indx_en_diff_coeff[ip]], lw=2.5, label='$p = \,$' + str("{:.2f}".format(p_grid[indx_en_diff_coeff[ip]])) + '$\, \mathrm{GeV}$')
    
plt.yscale('log')
plt.xlabel('$ r \in [0, \, R_{\mathrm{forward}}]$',fontsize=20)
plt.ylabel('$ D(p) \, [\mathrm{cm^2} \cdot \mathrm{s^{-1}}]$',fontsize=20)
plt.legend(fontsize=18, frameon=False, loc='best')
plt.tight_layout()
################################################################



plt.figure(figsize=(14, 5))

# diffusion coefficient subplot
plt.subplot(1, 2, 1)
plot_cosmetics_multi()


plt.loglog(p_grid, D_matrix[space_point_diff, :] / D_ref, lw=2.5, color='blue', label='$D(p) = D_0 \\cdot \\left( p \\big/ p_0 \\right)^\\delta$')
plt.xlabel('$ p \, [\mathrm{GeV}/c]$',fontsize=20)
plt.ylabel('$ D(p) \\bigg/ D_{\mathrm{ref}}$',fontsize=20)
plt.legend(fontsize=18, frameon=False)
plt.text(0.05, 0.73, '$D_0 = \,$' + str("${}$".format(f.format_data(D_0))) + '$\, \mathrm{cm^2} \cdot \mathrm{s}^{-1}$', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.05, 0.61, '$p_0 = \,$' + str("${}$".format(f.format_data(p_0_diff))) + ' GeV', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.05, 0.49, '$\\delta = \,$' + str("{:.2f}".format(delta_diff)), fontsize=18, transform = plt.gca().transAxes)
plt.text(0.25, 0.2, '$r = \,$' + str("{:.0f}".format(x_grid[space_point_diff] * L_ref)) + '\, pc', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.25, 0.1, '$D_{\mathrm{ref}} = \,$' + str("${}$".format(f.format_data(D_ref))) + '$\, \mathrm{cm^2} \cdot \mathrm{s}^{-1}$', fontsize=18, transform = plt.gca().transAxes)
################################################################


# velocity-profile subplot
plt.subplot(1, 2, 2)
plot_cosmetics_multi()


plt.plot(x_grid, v_adv, lw=2.5, color='blue', label=label_v_profile)
plt.axvline(x=x_grid[shock_index-1], ls=':', lw=2., color='green', label='$i^* - 1$')
plt.axvline(x=x_grid[shock_index+1], ls=':', lw=2., color='red', label='$i^* + 1$')
plt.xlabel('$ r \in [0, \, R_{\mathrm{forward}}]$',fontsize=20)
plt.ylabel('$ v_{_\mathrm{dimensionless}}$',fontsize=20)
plt.legend(fontsize=18, frameon=False, loc='upper right')


if case_velocity == 'symmetric_velocity':
    plt.text(0.05, 0.35, '$v_0 = \,$' + str("${}$".format(f.format_data(v_0))) + '$\, \mathrm{km} \cdot \mathrm{s}^{-1}$', fontsize=19, transform = plt.gca().transAxes)
    plt.text(0.05, 0.25, '$\\Delta x_{v} = \,$' + str("{:.0f}".format(DeltaL_v)) + '$\, \mathrm{pc}$', fontsize=19, transform = plt.gca().transAxes)
    plt.text(0.05, 0.15, '$r_{\mathrm{inj}} = L/2$', fontsize=19, transform = plt.gca().transAxes)
elif case_velocity == 'step_function':
    plt.text(0.50, 0.6, '$v_0 = \,$' + str("${}$".format(f.format_data(v_0))) + '$\, \mathrm{km} \cdot \mathrm{s}^{-1}$', fontsize=19, transform = plt.gca().transAxes)
    plt.text(0.50, 0.5, '$r_{\mathrm{inj}} = L/2$', fontsize=19, transform = plt.gca().transAxes)
else:
    plt.text(0.50, 0.6, '$v_0 = \,$' + str("${}$".format(f.format_data(v_0))) + '$\, \mathrm{km} \cdot \mathrm{s}^{-1}$', fontsize=18, transform = plt.gca().transAxes)
    plt.text(0.50, 0.5, '$r_{\mathrm{inj}} = \,$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=18, transform = plt.gca().transAxes)
    
    
plt.text(shock_location - 0.3*shock_location, 0.15, 'Upstream', fontsize=18, rotation=90, bbox=dict(lw=2, facecolor='none', edgecolor='red', pad=5.0), transform = plt.gca().transAxes)
plt.text(0.6, 0.15, 'Downstream', fontsize=18, bbox=dict(lw=2, facecolor='none', edgecolor='red', pad=5.0), transform = plt.gca().transAxes)
plt.tight_layout()
plt.savefig(dirName + 'VelocityProfile.pdf',format='pdf',bbox_inches='tight', dpi=200)
################################################################



# plot the magnetic field and plot the density
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()


plt.plot(x_grid, B_space, lw=2.5, color='blue', label='$B(r)$')
plt.axhline(y=B_TS, ls='--', lw=2., color='orange', label='$B_\mathrm{TS} = \;$' + str("{:.2f}".format(B_TS)) + '$\, \\mu \mathrm{G}$')
plt.axvline(x=L_c/L_ref, ls='--', lw=2., color='green', label='$L_c \equiv \mathrm{SC \; size}$')
plt.yscale('log')
plt.xlabel('$ r \in [0, \, R_{\mathrm{forward}}]$',fontsize=20)
plt.ylabel('$B \; [\mu \mathrm{G}]$',fontsize=20)
plt.legend(fontsize=18, frameon=False, loc='best')



plt.subplot(1, 2, 2)
plot_cosmetics_multi()
    

plt.plot(x_grid, wind_density_array, lw=2.5, color='blue', label='$\\rho_w = \\frac{\dot{M}}{4 \pi r^2 u_w}$')
plt.plot(x_grid, n_bubble_array, lw=2.5, color='red', label='$n_{\mathrm{bubble}}(r)$')
plt.yscale('log')
plt.axvline(x=L_c/L_ref, lw=2., ls='--', color='green', label='$L_c \equiv \mathrm{SC \; size}$')
plt.axvline(x=R_TS / L_ref, lw=2., ls='--', color='orange', label='$R_{\mathrm{TS}}$')
plt.xlabel('$ r \in [0, \, R_{\mathrm{forward}}]$',fontsize=20)
plt.ylabel('$\\rho_w \; [\mathrm{cm^{-3}}]$',fontsize=20)
plt.legend(fontsize=18, frameon=False, loc='best')
plt.tight_layout()
############################################



##################
# Injection data #
##################
p_injection = p_grid[0]*3      # p_inj at the shock, in [GeV]
indx_p_inj = np.argmin( abs(p_injection - p_grid) )
indx_x_inj = shock_index
x_injection = x_grid[indx_x_inj]

print('injection momentum =', p_injection, '[GeV]')
print('injection at p index:', indx_p_inj)



##########################################
### Block to study the source function ###
##########################################
# Choose a source function, as injection rate #
case = 'Shock'                 # other options include: gaussian_burst, constant, StepFunction, ...
p_0 = min(p_grid)
Source_slope = 8.
print('The shock is at index:', indx_x_inj)
print('')


# normalization of the injection function (norm_inj)
norm_inj = 1.
def source_func(x_space, p_momentum):
    
    if case == 'Shock':
        temp_p = (p_momentum[:] / p_injection)**(-Source_slope)
        temp_x = [norm_inj if ix == indx_x_inj or ix == indx_x_inj-1 else 0. for ix in range(len(x_grid))]
        
    source_function_array = np.asarray([[temp_x[ix] * temp_p[ip] for ip in range(len(p_grid))] for ix in range(len(x_grid))])
    return source_function_array


source_matrix = source_func(x_grid, p_grid)
source_matrix[source_matrix < 1.e-100] = 0
print(f'shape of the source matrix: {source_matrix.shape}')
print('')


# Plot the source function
plt.figure(figsize=(13,5))


plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.plot(x_grid, source_matrix[:, indx_p_inj], lw=2.5, color='blue')
plt.axvline(x=x_grid[indx_x_inj-1], ls='--', lw=2., color='green', label='$i^* - 1$')
plt.axvline(x=x_grid[indx_x_inj], ls='--', lw=2., color='orange', label='$i^*$')
plt.axvline(x=x_grid[indx_x_inj+1], ls='--', lw=2., color='red', label='$i^* + 1$')
plt.title('Injection rate: ' + str(case), fontsize = 20, pad=8)
plt.xlabel('$r \in [0, \, R_{\mathrm{forward}}]$', fontsize=20)
plt.ylabel('$f(r, p, t)$', fontsize=20)
plt.text(0.45, 0.2, '$p = \,$' + str("{:.1f}".format(p_grid[indx_p_inj])) + ' GeV', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.45, 0.1, '$(N_r, N_p) = ($' + str(Nx) + ',\,' + str(Np) + '$)$', fontsize=18, transform = plt.gca().transAxes)
plt.legend(frameon=False, fontsize=18)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

plt.axvline(x=p_grid[indx_p_inj], ls='--', lw=2.5, color='green', label='Injection momentum')
plt.loglog(p_grid, source_matrix[indx_x_inj, :], lw=2.5, color='blue', label='$p_{\mathrm{inj}} \\simeq \, $' + str(round(p_grid[indx_p_inj])) + '$\, \mathrm{GeV}$')
plt.title('Injection spectrum: ' + str(case), fontsize=20, pad=8)
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
plt.ylabel('$f(r, p, t)$', fontsize=20)
plt.legend(frameon=False, fontsize=18)
plt.text(0.35, 0.1, '$r = \, r_{\mathrm{shock}}$', fontsize=18, transform = plt.gca().transAxes)
plt.tight_layout()


plt.savefig(dirName + 'SourceFunction.pdf',format='pdf',bbox_inches='tight', dpi=200)



#####################################################################################
## Block to adjust the source function: medium point + multiplication by time-step ##
#####################################################################################
print('original source matrix:', source_matrix[indx_x_inj, :9])


Source_lower1 = [max(source_matrix[ix,:]) * (p_lower1 / p_0)**(-Source_slope) for ix in range(len(x_grid))]

source_matrix_adjusted = np.zeros( (len(x_grid), len(p_grid)), dtype=np.float64 )
for ix in range(len(x_grid)):
    source_matrix_adjusted[ix,:] = [(source_matrix[ix,ip] + Source_lower1[ix]) / 2. if ip == 0 else (source_matrix[ix,ip] + source_matrix[ix,ip-1]) / 2. for ip in range(len(p_grid))]
    
    
print('adjusted source matrix:', source_matrix_adjusted[indx_x_inj, :9])
print('')


plt.figure(figsize=(13,5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.plot(x_grid, source_matrix[:,indx_p_inj], lw=2.5, color='blue', label='original source')
plt.plot(x_grid, source_matrix_adjusted[:,indx_p_inj], lw=2.5, color='orange', label='adjusted source')
plt.xlabel('$r \in [0, \, R_{\mathrm{forward}}]$', fontsize=20)
plt.ylabel('$f(r, \\bar{p}, 0)$', fontsize=20)
plt.legend(frameon=False, fontsize=18, loc='best')


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

plt.scatter(p_lower1, Source_lower1[indx_x_inj], label='extrapolated low point')
plt.loglog(p_grid[:], source_matrix[indx_x_inj, :], lw=2.5, color='blue', label='original source')
plt.loglog(p_grid[:], source_matrix_adjusted[indx_x_inj, :], lw=2.5, color='orange', label='adjusted source')
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
plt.ylabel('$f(\\bar{r}, p, 0)$', fontsize=20)
plt.legend(frameon=False, fontsize=18, loc='best')
plt.tight_layout()



#################################
## Block to define the mapping ##
#################################
def indx_mapping(ix_, ip_):    
    return ix_ + len(x_grid) * ip_ 



source_mapped = np.zeros( len(x_grid)*len(p_grid) + 1, dtype=np.float64 )
for ix in range(0, len(x_grid)):
    for ip in range(0, len(p_grid)):
        source_mapped[indx_mapping(ix,ip)] = source_matrix_adjusted[ix,ip]
        
        
D_mapped = np.zeros( len(x_grid)*len(p_grid), dtype=np.float64 )
for ix in range(0, len(x_grid)):
    for ip in range(0, len(p_grid)):
        D_mapped[indx_mapping(ix,ip)] = D_matrix_dless[ix,ip]



def rev_dict(k):
    i = int(k % len(x_grid))
    j = int((k - i) / len(x_grid))
    return i, j
        

#print('')
#for ik in range(0, len(x_grid)*len(p_grid)):
#    print(f'k = {ik}   ->   {rev_dict(ik)}')



#####################
## Transport cases ##
#####################
case_transport = 'DSA_spherical'
#particle_type = 'protons_ppLoss'
particle_type = 'protons'
print(f'I am considering the case: {case_transport} for {particle_type} cosmic rays.')


if case_transport == 'DSA_spherical':
    
    if particle_type == 'protons':

        # without losses
        def fun_rhs(t, f_):

            dfdt = np.zeros( len(x_grid)*len(p_grid), dtype=np.float64 )
            for ik in range(1, len(x_grid)*len(p_grid) - 1 ):

                ix, ip = rev_dict(ik)
                f_[indx_mapping(0,ip)] = (D_matrix_dless[0,ip] / x_grid[1]) / (D_matrix_dless[0,ip] / x_grid[1] + v_adv[0]) * f_[indx_mapping(1,ip)]

                if ix != 0 and ix != len(x_grid)-1 and ip != 0 and ip != len(p_grid)-1 :


                    dfdt[ik] = 1. / ( 2. * x_grid[ix]**2 * (x_grid[ix+1] - x_grid[ix-1]) ) \
                           * ( ( x_grid[ix]**2 + x_grid[ix+1]**2 ) * (D_matrix_dless[ix,ip] + D_matrix_dless[ix+1,ip]) * (f_[indx_mapping(ix+1,ip)] - f_[indx_mapping(ix,ip)])/(x_grid[ix+1] - x_grid[ix]) \
                           - ( x_grid[ix]**2 + x_grid[ix-1]**2 ) * (D_matrix_dless[ix,ip] + D_matrix_dless[ix-1,ip]) * (f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix-1,ip)])/(x_grid[ix] - x_grid[ix-1]) ) \
                           - v_adv[ix] * (f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix-1,ip)]) / (x_grid[ix] - x_grid[ix-1]) \
                           + (2. * v_adv[ix] / x_grid[ix]) * p_grid[ip] / 3. * ( f_[indx_mapping(ix,ip+1)] - f_[indx_mapping(ix,ip)] ) / (p_grid[ip+1] - p_grid[ip]) \
                           + p_grid[ip] / 3. * v_adv_derivative[ix] * ( f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix,ip-1)] ) / (p_grid[ip] - p_grid[ip-1]) \
                           + source_matrix_adjusted[ix,ip]

            return dfdt
        
        
    elif particle_type == 'protons_ppLoss':

        # with pion-production losses
        def fun_rhs(t, f_):

            dfdt = np.zeros( len(x_grid)*len(p_grid), dtype=np.float64 )
            for ik in range(1, len(x_grid)*len(p_grid) - 1 ):

                ix, ip = rev_dict(ik)
                f_[indx_mapping(0,ip)] = (D_matrix_dless[0,ip] / x_grid[1]) / (D_matrix_dless[0,ip] / x_grid[1] + v_adv[0]) * f_[indx_mapping(1,ip)]

                if ix != 0 and ix != len(x_grid)-1 and ip != 0 and ip != len(p_grid)-1 :


                    dfdt[ik] = 1. / ( 2. * x_grid[ix]**2 * (x_grid[ix+1] - x_grid[ix-1]) ) \
                           * ( ( x_grid[ix]**2 + x_grid[ix+1]**2 ) * (D_matrix_dless[ix,ip] + D_matrix_dless[ix+1,ip]) * (f_[indx_mapping(ix+1,ip)] - f_[indx_mapping(ix,ip)])/(x_grid[ix+1] - x_grid[ix]) \
                           - ( x_grid[ix]**2 + x_grid[ix-1]**2 ) * (D_matrix_dless[ix,ip] + D_matrix_dless[ix-1,ip]) * (f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix-1,ip)])/(x_grid[ix] - x_grid[ix-1]) ) \
                           - v_adv[ix] * (f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix-1,ip)]) / (x_grid[ix] - x_grid[ix-1]) \
                           + (2. * v_adv[ix] / x_grid[ix]) * p_grid[ip] / 3. * ( f_[indx_mapping(ix,ip+1)] - f_[indx_mapping(ix,ip)] ) / (p_grid[ip+1] - p_grid[ip]) \
                           + p_grid[ip] / 3. * v_adv_derivative[ix] * ( f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix,ip-1)] ) / (p_grid[ip] - p_grid[ip-1]) \
                           - 1. / p_grid[ip]**2 * (loss_rate_pp_collision_matrix[ix, ip+1] * p_grid[ip+1]**2 * f_[indx_mapping(ix,ip+1)] - loss_rate_pp_collision_matrix[ix, ip] * p_grid[ip]**2 * f_[indx_mapping(ix,ip)]) / ( pdot_ref * (p_grid[ip+1] - p_grid[ip]) ) \
                           + source_matrix_adjusted[ix,ip]

            return dfdt
        
        
    elif particle_type == 'leptons':

        # with IC-Synchrotron losses
        def fun_rhs(t, f_):

            dfdt = np.zeros( len(x_grid)*len(p_grid), dtype=np.float64 )
            for ik in range(1, len(x_grid)*len(p_grid) - 1 ):

                ix, ip = rev_dict(ik)
                f_[indx_mapping(0,ip)] = (D_matrix_dless[0,ip] / x_grid[1]) / (D_matrix_dless[0,ip] / x_grid[1] + v_adv[0]) * f_[indx_mapping(1,ip)]

                if ix != 0 and ix != len(x_grid)-1 and ip != 0 and ip != len(p_grid)-1 :


                    dfdt[ik] = 1. / ( 2. * x_grid[ix]**2 * (x_grid[ix+1] - x_grid[ix-1]) ) \
                           * ( ( x_grid[ix]**2 + x_grid[ix+1]**2 ) * (D_matrix_dless[ix,ip] + D_matrix_dless[ix+1,ip]) * (f_[indx_mapping(ix+1,ip)] - f_[indx_mapping(ix,ip)])/(x_grid[ix+1] - x_grid[ix]) \
                           - ( x_grid[ix]**2 + x_grid[ix-1]**2 ) * (D_matrix_dless[ix,ip] + D_matrix_dless[ix-1,ip]) * (f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix-1,ip)])/(x_grid[ix] - x_grid[ix-1]) ) \
                           - v_adv[ix] * (f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix-1,ip)]) / (x_grid[ix] - x_grid[ix-1]) \
                           + (2. * v_adv[ix] / x_grid[ix]) * p_grid[ip] / 3. * ( f_[indx_mapping(ix,ip+1)] - f_[indx_mapping(ix,ip)] ) / (p_grid[ip+1] - p_grid[ip]) \
                           + p_grid[ip] / 3. * v_adv_derivative[ix] * ( f_[indx_mapping(ix,ip)] - f_[indx_mapping(ix,ip-1)] ) / (p_grid[ip] - p_grid[ip-1]) \
                           - 1. / p_grid[ip]**2 * (loss_rate_KN_matrix[ix, ip+1] * p_grid[ip+1]**2 * f_[indx_mapping(ix,ip+1)] - loss_rate_KN_matrix[ix, ip] * p_grid[ip]**2 * f_[indx_mapping(ix,ip)]) / ( pdot_ref * (p_grid[ip+1] - p_grid[ip]) ) \
                           + source_matrix_adjusted[ix,ip]

            return dfdt



#########################################
## Define the sparsity of the jacobian ##
#########################################
from scipy.sparse import dia_array

Nk = len(x_grid) * len(p_grid)
data = [np.ones( Nk, dtype=np.float64 )]*5
jac_sparsity = dia_array( (data, [-len(x_grid), -1, 0, 1, len(x_grid)]), shape=(Nk, Nk), dtype=np.float64 )



########################################################
## Modify the solve_ivp method to show a progress bar ##
########################################################
from scipy.integrate._ivp.base import OdeSolver
from tqdm import tqdm


# save the old methods
old_init = OdeSolver.__init__
old_step = OdeSolver.step

# define our own methods
def new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):

    # define the progress bar
    self.pbar = tqdm(total=t_bound - t0, unit='ut', initial=t0, ascii=True, desc='IVP')
    self.last_t = t0
    
    # call the old method - we still want to do the old things too!
    old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


def new_step(self):
    # call the old method
    old_step(self)
    
    # update the bar
    tst = self.t - self.last_t
    self.pbar.update(tst)
    self.last_t = self.t

    # close the bar if the end is reached
    if self.t >= self.t_bound:
        self.pbar.close()


# overwrite the old methods with our customized ones
OdeSolver.__init__ = new_init
OdeSolver.step = new_step



################################
## Block with the calculation ##
################################
last_timestep_dless = ( t_run_yr*conv_yr_sec ) / Tbar_dim_s
t_eval_array = np.logspace(start=np.log10( min( timescale_acc_sec ) / Tbar_dim_s ), stop=np.log10( last_timestep_dless - 0.001*last_timestep_dless ), num=7)
start = time.process_time()

output = scipy.integrate.solve_ivp(fun_rhs, [ min(t_eval_array), max(t_eval_array)], np.zeros( len(x_grid)*len(p_grid), dtype=np.float64 ), method='BDF', t_eval=t_eval_array, jac_sparsity=jac_sparsity, dense_output=False, events=None, vectorized=False, args=None, atol=1.e-2, rtol=1.e-2)

runtime = time.process_time() - start
print(f'It took {runtime} seconds = {runtime / 3600} hours to find the solution.')
print('')



import psutil
import sys
from sys import getsizeof
import humanize

print('*************')
print('** Results **')
print('*************')
print('')
print(f'len(x_grid) = {len(x_grid)}, len(p_grid) = {len(p_grid)}, Nk = {Nk}, case = {case_transport}')
print(f'size of my array: {sys.getsizeof(np.arange(Nk*5)) / 2**30} [GiB] = {sys.getsizeof(np.arange(Nk*5)) / 2**30 * 1.07374} [GB]')
print('')
print('MEMORY USAGE')
print('')
print(f'total memory (including SWAP) = {humanize.naturalsize(psutil.virtual_memory()[0])}')
print(f'available memory for processes = {humanize.naturalsize(psutil.virtual_memory()[1])}')
print(f'RAM memory % used: = {psutil.virtual_memory()[2]} %')
print(f'RAM Used: {humanize.naturalsize(psutil.virtual_memory()[3])}')
print(f'memory not used and is readily available: {humanize.naturalsize(psutil.virtual_memory()[4])}')
print(f'active = {humanize.naturalsize(psutil.virtual_memory()[5])}')
print(f'inactive = {humanize.naturalsize(psutil.virtual_memory()[6])}')
print(f'wired = {humanize.naturalsize(psutil.virtual_memory()[7])}')
print('')



#############################
## Time evolving snapshots ##
#############################
print(f'last dimensionless instant: {t_run_yr*conv_yr_sec / Tbar_dim_s}')
print(f'length of the time array: {len(output.t)}')
print(f'time steps: {output.t}')
print('')


sol = np.zeros( (len(output.t), len(x_grid), len(p_grid)), dtype=np.float64 )
for it in range(0, len(output.t)):
    for ip in range(0, len(p_grid)):

        momentum_indx_start = ip * len(x_grid)
        momentum_indx_stop = (ip+1) * len(x_grid) - 1
        
        sol[it, :, ip] = output.y[momentum_indx_start:momentum_indx_stop+1, it]
        sol[it, 0, ip] = (D_mapped[indx_mapping(0,ip)] / x_grid[1]) / (D_mapped[indx_mapping(0,ip)] / x_grid[1] + v_adv[0]) * sol[it, 1, ip]

    
plt.figure(figsize=(13,5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()


for it in range(0, len(output.t)):
    if it == 0:
        plt.plot(x_grid, sol[it, :, indx_p_inj], ls='-', lw=2.5, color='blue', label='$t = \,$' + str(it))
    else:
        plt.plot(x_grid, sol[it, :, indx_p_inj], ls='-', lw=2., label='$t = \,$' + str(it))

            
plt.title('$T_{\mathrm{run}} = \,$' + str("{:.3f}".format(t_run_Myr)) + '$\; \mathrm{Myr}, \; \; p \simeq \,' + str(round(p_grid[indx_p_inj])) + '\, \mathrm{GeV}$', fontsize=18, pad=8)
plt.axvline(x=x_grid[indx_x_inj], ls=':', lw=2., color='green', label='$i^*$')
plt.axhline(y=0., ls='--', lw=1.5, color='red')
plt.xlabel('$r \in [0, \, R_{\mathrm{forward}}]$', fontsize=20)
plt.ylabel('$f(r, T_{\mathrm{run}})$', fontsize=20)
plt.legend(frameon=False, fontsize=16, loc='best')



plt.subplot(1, 2, 2)
plot_cosmetics_multi()


indx_x_spectrum = indx_x_inj
indx_10TeV = np.argmin( abs(p_grid - 1.e4) )


for it in range(0, len(output.t)):
    if it == 0:
        plt.loglog(p_grid[:], sol[it, indx_x_inj, :], ls='-', lw=2.5, color='blue', label='$t = \,$' + str(it))
    else:
        plt.loglog(p_grid[:], sol[it, indx_x_inj, :], ls='-', lw=2., label='$t = \,$' + str(it))



plt.title('$T_{\mathrm{run}} = \,$' + str("{:.3f}".format(t_run_Myr)) + '$\; \mathrm{Myr}, \;\; r = \; $' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=18, pad=8)
if particle_type == 'leptons':
    plt.axvline(x=p_grid[indx_intersec_LossAcc], ls='--', lw=1.5, color='Orange', label='$\\tau_{\mathrm{loss}} \leq \\tau_{\mathrm{acc}}$')
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
plt.ylabel('$f(p,T_{\mathrm{run}})$', fontsize=20)
plt.legend(frameon=False, fontsize=16, loc='best')
plt.tight_layout()



#####################################
## Block to plot the last snapshot ##
#####################################
# Fit the propagated function #
def fit_func(p_fit, a, b):
    return (norm_inj*a) * (p_fit / min(p_grid))**(-b)

factor_aboveInj_fit_in = 2.
factor_aboveInj_fit_fin = 5.
p_fit_array = np.logspace(start=np.log10(p_grid[indx_p_inj]*factor_aboveInj_fit_in), stop=np.log10(p_grid[indx_p_inj]*factor_aboveInj_fit_fin), num=5)
p_fit_array_indx = [np.argmin( abs(p_fit_array[i] - p_grid) ) for i in range(len(p_fit_array))]


popt, pcov = optimize.curve_fit(fit_func, p_grid[p_fit_array_indx], sol[-1, indx_x_inj, p_fit_array_indx], maxfev=1000)
propagated_slope_fit = popt[1]
print('Fit`s parameters for the final solution (a, b):', popt)
print('')
###############################



plt.figure(figsize=(13,5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()


plt.plot(x_grid, sol[-1, :, indx_p_inj], ls='-', lw=2.5, color='blue', label='num')
plt.title('$T_{\mathrm{run}} = \,$' + str("{:.3f}".format(t_run_Myr)) + '$\; \mathrm{Myr}, \; \; p \simeq \,' + str(round(p_grid[indx_p_inj])) + '\, \mathrm{GeV}$', fontsize=18, pad=8)
plt.xlabel('$r \in [0, \, R_{\mathrm{forward}}]$', fontsize=20)
plt.ylabel('$f(r,T_{\mathrm{run}})$', fontsize=20)
plt.legend(frameon=False, fontsize=20, loc='upper left')
plt.text(0.2, 0.1, '$(N_r, N_p) = ($' + str(Nx) + ',\,' + str(Np) + '$)$', fontsize=16, transform = plt.gca().transAxes)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()


indx_x_spectrum = indx_x_inj
indx_10TeV = np.argmin( abs(p_grid - 1.e4) )


plt.loglog(p_grid[:], sol[-1, indx_x_inj, :], ls='-', lw=2.5, color='blue', label='num')
plt.loglog(p_grid, fit_func(p_grid, popt[0], popt[1]), lw=2., ls='--', color='red', label='Fit: $\\gamma = \,$' + str("{:.2f}".format(popt[1])))
plt.scatter(p_fit_array, sol[-1, indx_x_inj, p_fit_array_indx])
plt.title('$T_{\mathrm{run}} = \,$' + str("{:.3f}".format(t_run_Myr)) + '$\; \mathrm{Myr}, \;\; r = \;$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=18, pad=8)
plt.axvline(x=p_grid[indx_intersec_tauAcc_upstream], ls='--', color='magenta', label='$D \\big/u_{\mathrm{up}} = \\Delta r_{\mathrm{up}}$')
if particle_type == 'leptons':
    plt.axvline(x=p_grid[indx_intersec_LossAcc], ls='--', lw=1.5, color='Orange', label='$\\tau_{\mathrm{loss}} \leq \\tau_{\mathrm{acc}}$')
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
plt.ylabel('$f(p,T_{\mathrm{run}})$', fontsize=20)
plt.legend(frameon=False, fontsize=20, loc='best')
plt.tight_layout()

plt.savefig(dirName + 'LastSnapshot_deltadiff=' + str(delta_diff) + '.pdf',format='pdf',bbox_inches='tight', dpi=200)



############################################################
## Block to extend the numerical solution to low energies ##
############################################################
# number of low-E points, according to the log factor 'factor grid' computed above
factor_grid = [p_grid[i] / p_grid[i-1] for i in range(1, len(p_grid))]
Np_lowE = round(np.emath.logn( factor_grid[0], ( (p_grid[indx_p_inj]*factor_aboveInj_fit_fin) / 1.) ))
p_lowE = np.logspace(start=0., stop=np.log10( p_grid[indx_p_inj]*factor_aboveInj_fit_fin ), num=Np_lowE)
indx_pFit_in = np.argmin( abs( p_grid - p_grid[indx_p_inj]*factor_aboveInj_fit_in ) )
indx_pFit_fin = np.argmin( abs( p_grid - p_grid[indx_p_inj]*factor_aboveInj_fit_fin ) )
indx_pFit_medium = round( (indx_pFit_in + indx_pFit_fin) / 2 )

p_array_integral_tot = np.concatenate( (p_lowE, p_grid[indx_pFit_fin:]) )


# Fit the propagated function #
def fit_locations(p_fit, a, b):
    return a * (p_fit / p_grid[0])**(-b)

p_fit_loc = np.logspace(start=np.log10(p_grid[indx_p_inj]*factor_aboveInj_fit_in), stop=np.log10(p_grid[indx_p_inj]*factor_aboveInj_fit_fin), num=5)
p_fit_loc_indx = [np.argmin( abs(p_fit_loc[i] - p_grid) ) for i in range(len(p_fit_loc))]
###############################


slope_loc = np.zeros( len(x_grid), dtype=np.float64 )
for ix in range(len(x_grid)):
    slope_loc_temp, slope_cov = optimize.curve_fit(fit_locations, p_grid[p_fit_loc_indx], sol[-1, ix, p_fit_loc_indx], maxfev=1000)
    slope_loc[ix] = slope_loc_temp[1]

    
CR_spectrum = np.zeros( (len(x_grid), len(p_array_integral_tot)), dtype=np.float64 )
for ix in range(len(x_grid)-1):
    if ix < shock_index:
        CR_spectrum[ix, :] = np.concatenate( (sol[-1, ix, indx_pFit_fin] * ( p_lowE/p_grid[indx_pFit_fin] )**( - slope_loc[ix] ), sol[-1, ix, indx_pFit_fin:] ) )
    else:
        Amplitude_break = sol[-1, ix, indx_pFit_in]
        SBPL_extrapolation = models.SmoothlyBrokenPowerLaw1D(amplitude=Amplitude_break, x_break=p_grid[indx_pFit_in], alpha_1=4.00, alpha_2=slope_loc[ix], delta=1.)
        SBPL_extrapolation_array = SBPL_extrapolation(p_lowE)
        CR_spectrum[ix, :] = np.concatenate( (SBPL_extrapolation_array, SBPL_extrapolation_array[-1]/sol[-1, ix, indx_pFit_fin] * sol[-1, ix, indx_pFit_fin:] ) )



loc_spectrum_upstream = np.linspace(start=start_x_grid, stop=shock_location, num=5, endpoint=False)
loc_spectrum_downstream = np.linspace(start=x_grid[shock_index], stop=x_grid[-2], num=6, endpoint=True)
loc_spectrum = np.concatenate( (loc_spectrum_upstream, loc_spectrum_downstream) )
indx_spectrum = [np.argmin( abs(loc_spectrum[ix] - x_grid) ) for ix in range(len(loc_spectrum))]

print(f'shock at index {shock_index}, location = {shock_location} = {shock_location*L_ref} [pc]')
print('')
print(f'dimensionless locations where I am plotting the spectrum: \n{[x_grid[indx_spectrum[i]] for i in range(len(loc_spectrum))]}')
print(f'dimensional locations where I am plotting the spectrum: \n{[x_grid[indx_spectrum[i]]*L_ref for i in range(len(loc_spectrum))]} [pc]')
print('')


CR_spectrum_shock = CR_spectrum[indx_x_inj, :]
integral_CR_pressure = np.trapz( ( p_array_integral_tot/p_grid[0] )**3. * CR_spectrum_shock, p_array_integral_tot/p_grid[0] )

    
# factor to normalize the CR distribution function
xi_CR = 0.18
CR_norm = 3 * xi_CR * ram_pressure_TS_GeV / (4 * np.pi * p_grid[0]**4 * integral_CR_pressure)


plt.figure(figsize=(13,5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.title('$T_{\mathrm{run}} = \,$' + str("{:.3f}".format(t_run_Myr)) + '$\; \mathrm{Myr}, \;\; r = \;$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=15, pad=8)
plt.loglog(p_array_integral_tot, CR_norm*CR_spectrum_shock, lw=2., color='blue', label='low-E extension')
plt.loglog(p_grid, CR_norm*sol[-1, indx_x_inj, :], ls='--', lw=2., color='red', label='num sol, $s=-$' + str("{:.2f}".format( slope_loc[indx_x_inj] )))
plt.axvline(x=p_grid[indx_p_inj], ls=':', color='cyan', label='$p_{\mathrm{inj}} \simeq \,$' + str(round(p_grid[indx_p_inj])) + '$\, \mathrm{GeV}$')
plt.axvline(x=p_grid[indx_pFit_medium], ls='--', color='orange', label='$p_{\mathrm{extrapol}} \simeq \,$' + str(round(p_grid[indx_pFit_medium])) + '$\, \mathrm{GeV}$')
plt.axvline(x=p_grid[indx_pFit_fin], ls='--', color='magenta')
plt.axvline(x=p_grid[indx_pFit_in], ls='--', color='green')
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=16)
plt.ylabel('$f(p) \; [\mathrm{GeV^{-3}} \cdot \mathrm{cm^{-3}}]$', fontsize=16)
plt.legend(fontsize=15, frameon=False, loc='upper right')


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

for ix in range(len(loc_spectrum)):
    if loc_spectrum[ix] > x_grid[indx_x_inj]:
        plt.loglog(p_array_integral_tot, p_array_integral_tot**4 * CR_norm*CR_spectrum[indx_spectrum[ix],:], lw=2.5, label='$r \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')

plt.title('Downstream extrapolated spectra', fontsize=15, pad=8)
plt.axvline(x=p_grid[indx_pFit_medium], ls='--', lw=1.5, color='orange', label='$p_{\mathrm{extrapol}} \simeq \,$' + str(round(p_grid[indx_pFit_medium])) + '$\, \mathrm{GeV}$')
plt.ylim( np.max( p_array_integral_tot[:]**4 * CR_norm*CR_spectrum[indx_x_inj,:] ) / 1.e5, np.max( p_array_integral_tot[:]**4 * CR_norm*CR_spectrum[indx_x_inj,:]) * 1.e1)
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=16)
plt.ylabel('$p^4 \cdot f(p) \; [\mathrm{GeV} \cdot \mathrm{cm^{-3}}]$', fontsize=16)
plt.legend(fontsize=15, frameon=False, loc='best')
plt.tight_layout()


print(f'ram pressure of the wind: {ram_pressure_TS_GeV} [GeV cm^(-3)]')
print(f'minimum momentum considered in the run: {p_grid[0]} [GeV]')
print(f'dimensionless integral of the CR pressure = {integral_CR_pressure}')
print(f'normalization of the CR distribution = {CR_norm} [GeV^(-3) cm^(-3)]')



#################################################################
## Block to plot the last snapshot, solution multiplied by p^4 ##
#################################################################
indx_x_spectrum = indx_x_inj
indx_1TeV = np.argmin( abs(p_grid - 1.e3) )
indx_10TeV = np.argmin( abs(p_grid - 1.e4) )


plt.figure(figsize=(13,5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.plot(x_grid, CR_norm*sol[-1, :, indx_p_inj], ls='-', lw=2.5, color='blue', label='num')         
plt.title('$T_{\mathrm{run}} = \,$' + str("{:.3f}".format(t_run_Myr)) + '$\; \mathrm{Myr}, \; \; p \simeq \,' + str(round(p_grid[indx_p_inj])) + '\, \mathrm{GeV}$', fontsize=18, pad=8)
plt.xlabel('$r \in [0, \, R_{\mathrm{forward}}]$', fontsize=20)
plt.ylabel('$f(r,T_{\mathrm{run}}) \; [\mathrm{GeV^{-3}} \cdot \mathrm{cm^{-3}}]$', fontsize=20)
plt.legend(frameon=False, fontsize=18, loc='upper left')
plt.text(0.2, 0.1, '$(N_r, N_p) = ($' + str(Nx) + ',\,' + str(Np) + '$)$', fontsize=16, transform = plt.gca().transAxes)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

plt.loglog(p_grid[indx_p_inj:], (p_grid[indx_p_inj:] / p_grid[0])**(4.) * CR_norm*sol[-1, indx_x_inj, indx_p_inj:], ls='-', lw=2.5, color='blue', label='num, $\\gamma = \,$' + str("{:.2f}".format(popt[1])))
plt.xlim(p_grid[indx_p_inj])
plt.ylim(np.max( (p_grid[indx_p_inj:] / p_grid[0])**(4.) * CR_norm*sol[-1, :, indx_p_inj:] ) / 1.e4, np.max( (p_grid[indx_p_inj:] / p_grid[0])**(4.) * CR_norm*sol[-1, :, indx_p_inj:] ) * 1.e1)
plt.title('$T_{\mathrm{run}} = \,$' + str("{:.3f}".format(t_run_Myr)) + '$\; \mathrm{Myr}, \; p_{\mathrm{inj}} = \,$' + str(round(p_grid[indx_p_inj])/1.e3) + '$\, \mathrm{TeV}$', fontsize=18, pad=8)
plt.axhline(y=np.max( (p_grid[indx_p_inj:] / p_grid[0])**(4.) * CR_norm*sol[-1, :, indx_p_inj:] ), ls='--', lw=2., color='green', label='$\gamma_{\mathrm{DSA}} = -\, 4$')
plt.axhline(y=np.max( (p_grid[indx_p_inj:] / p_grid[0])**(4.) * CR_norm*sol[-1, :, indx_p_inj:] ) / np.exp(1), ls='--', lw=2., color='red', label='$1 \\big/ e$')
plt.axvline(x=p_grid[indx_intersec_tauAcc_upstream], ls='--', color='magenta', label='$D \\big/u_{\mathrm{up}} = \\Delta r_{\mathrm{up}}$')
if particle_type == 'leptons':
    plt.axvline(x=p_grid[indx_intersec_LossAcc], ls='--', lw=2., color='Orange', label='$\\tau_{\mathrm{loss}} \leq \\tau_{\mathrm{acc}}$')
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
plt.ylabel('$p^4 \cdot f(p,T_{\mathrm{run}}) \; [\mathrm{GeV} \cdot \mathrm{cm^{-3}}]$', fontsize=20)
plt.text(0.05, 0.3, '$r = \;$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=16, transform = plt.gca().transAxes)
plt.text(0.05, 0.2, '$\\delta = \;$' + str(delta_diff), fontsize=16, transform = plt.gca().transAxes)
plt.text(0.05, 0.1, '$u_{w, \mathrm{max}} = \;$' + str("${}$".format(f.format_data(v_0))) + '$\, \mathrm{km \cdot s^{-1}}$', fontsize=16, transform = plt.gca().transAxes)
plt.legend(frameon=False, fontsize=16, loc='upper right')
plt.tight_layout()

plt.savefig(dirName + 'LastSnapshot_p4_deltadiff=' + str(delta_diff) + '.pdf',format='pdf',bbox_inches='tight', dpi=200)



##############################################################
## Block to plot the spatial solution at different energies ##
##############################################################
plt.figure(figsize=(7.5, 5.5))
plot_cosmetics_single()



p_chosen_array = np.logspace(start=np.log10(p_grid[indx_p_inj]), stop=np.log10(p_grid[-1]-0.1*p_grid[-1]), num=6)
indices_p_chosen_array = [np.argmin( (abs(p_grid - p_chosen_array[ip])) ) for ip in range(len(p_chosen_array))]
multiplication_factor = [max(sol[-1, :, indices_p_chosen_array[0]]) / max(sol[-1, :, indices_p_chosen_array[ip+1]])  for ip in range(len(p_chosen_array)-1)]

 
for ip in range(len(p_chosen_array)):
    if ip == 0:
        plt.plot(x_grid*L_ref, CR_norm*sol[-1, :, indices_p_chosen_array[ip]], lw=2.5, label='$p \simeq \,$' + str( round(p_grid[indices_p_chosen_array[ip]]) / 1.e3 ) + '$\, \mathrm{TeV}$')
    else:
        plt.plot(x_grid*L_ref, multiplication_factor[ip-1]*CR_norm*sol[-1, :, indices_p_chosen_array[ip]], lw=2.5, label='$p \simeq \,$' + str( round(p_grid[indices_p_chosen_array[ip]]) / 1.e3 ) + '$\, \mathrm{TeV}$')
    
    
plt.axhline(y=0., ls='--', lw=1.5, color='red')
plt.title('Spatial solution at different energies, \;$\\delta = \,$' + str(delta_diff), fontsize=20, pad=8)
plt.xlabel('$R \, [\mathrm{pc}]$', fontsize=20)
plt.ylabel('$f(r, T_{\mathrm{run}}) \; [\mathrm{GeV^{-3}} \cdot \mathrm{cm^{-3}}]$', fontsize=20)
plt.legend(fontsize=16, frameon=False)

plt.savefig(dirName + 'SpatialSol_DifferentEnergies_delta=' + str(delta_diff) + '.pdf',format='pdf',bbox_inches='tight', dpi=200)


#######################################################
## Block to plot the spectrum at different locations ##
#######################################################
plt.figure(figsize=(14,5.5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()

for ix in range(len(loc_spectrum)):
    if loc_spectrum[ix] <= shock_location:
        if abs(loc_spectrum[ix] - shock_location) <= 1.e-2:
            plt.loglog(p_grid[:], (p_grid[:] / p_grid[0])**propagated_slope_fit * sol[-1, indx_spectrum[ix], :], ls='-', lw=2.5, color='blue', label='$r_{\mathrm{shock}} \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')
        else:
            plt.loglog(p_grid[:], (p_grid[:] / p_grid[0])**propagated_slope_fit * sol[-1, indx_spectrum[ix], :], ls='-', lw=1.5, label='$r \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')


plt.xlim(p_grid[indx_p_inj])
plt.ylim(np.max((p_grid[:] / p_grid[0])**propagated_slope_fit * sol[-1, :, :]) / 1.e5, np.max((p_grid[:]/p_grid[0])**propagated_slope_fit * sol[-1, :, :]) * 1.e1)
plt.title('Upstream spectra', fontsize=20, pad=8)
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
plt.ylabel('$p^s \cdot f(p,T_{\mathrm{run}})$', fontsize=20)
plt.text(0.35, 0.15, '$s = \,$' + str("{:.2f}".format(propagated_slope_fit)), fontsize=18, transform = plt.gca().transAxes)
plt.text(0.35, 0.05, '$r_{\mathrm{shock}} = \,$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=18, transform = plt.gca().transAxes)
plt.legend(fontsize=16, frameon=False)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

for ix in range(len(loc_spectrum)):
    if loc_spectrum[ix] >= shock_location:
        if abs(loc_spectrum[ix] - shock_location) <= 1.e-2:
            plt.loglog(p_grid[:], (p_grid[:] / p_grid[0])**propagated_slope_fit * sol[-1, indx_spectrum[ix], :], ls='-', lw=2.5, color='blue', label='$r_{\mathrm{shock}} \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')
        else:
            plt.loglog(p_grid[:], (p_grid[:] / p_grid[0])**propagated_slope_fit * sol[-1, indx_spectrum[ix], :], ls='-', lw=1.5, label='$r \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')


plt.xlim(p_grid[indx_p_inj])
plt.ylim(np.max((p_grid[:] / p_grid[0])**propagated_slope_fit * sol[-1, :, :]) / 1.e5, np.max((p_grid[:]/p_grid[0])**propagated_slope_fit * sol[-1, :, :]) * 1.e1)
plt.title('Downstream spectra', fontsize=20, pad=8)
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
#plt.ylabel('$p^s \cdot f(p,T_{\mathrm{run}})$', fontsize=20)
plt.text(0.35, 0.15, '$s = \,$' + str("{:.2f}".format(propagated_slope_fit)), fontsize=18, transform = plt.gca().transAxes)
plt.text(0.35, 0.05, '$r_{\mathrm{shock}} = \,$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=18, transform = plt.gca().transAxes)
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()


plt.savefig(dirName + 'Spectrum_DifferentLocations_delta=' + str(delta_diff) + '.pdf',format='pdf',bbox_inches='tight', dpi=200)



#############################################
## Block to save the results in text files ##
#############################################
plotting_folder = '/Users/ottaviofornieri/PHYSICS_projects/GitProjects/StellarClusters/Stored_solutions/Plotting_Folder/'
internal_folder = plotting_folder + 'Np=' + str(Np) + '/'


try:
    # Create target Directory
    os.mkdir(internal_folder)
    print("Directory", internal_folder, "created")
    print("")
except FileExistsError:
    print("Directory", internal_folder, "already exists")
    print("")

    
    
for ix in range(len(loc_spectrum)):
    
    if delta_diff == 0.:
        
        if ix == 0:
            try:
                # Create target Directory
                os.mkdir(internal_folder + 'D-const/')
                print("Directory", internal_folder + 'D-const/', "created")
                print("")
            except FileExistsError:
                print("Directory", internal_folder + 'D-const/', "already exists")
                print("")
        
        data_txt_file = np.zeros( (1, len(p_grid)), dtype=np.float64 )
        data_txt_file[0,:] = sol[-1, indx_spectrum[ix], :]
        if abs(loc_spectrum[ix] - shock_location) <= 1.e-2:
            np.savetxt(internal_folder + 'D-const/t-final_D-const_x=Shock.txt', data_txt_file)
            np.savetxt(dirName + 't-final_D-const_x=Shock.txt', data_txt_file)
        else:
            np.savetxt(internal_folder + 'D-const/t-final_D-const_x=' + str( round(loc_spectrum[ix]*L_ref) ) + 'pc.txt', data_txt_file)
            np.savetxt(dirName + 't-final_D-const_x=' + str( round(loc_spectrum[ix]*L_ref) ) + 'pc.txt', data_txt_file)

            
            
    elif delta_diff == 0.33:
        
        if ix == 0:
            try:
                # Create target Directory
                os.mkdir(internal_folder + 'D-Kol/')
                print("Directory", internal_folder + 'D-Kol/', "created")
                print("")
            except FileExistsError:
                print("Directory", internal_folder + 'D-Kol/', "already exists")
                print("")
        
        data_txt_file = np.zeros( (1, len(p_grid)), dtype=np.float64 )
        data_txt_file[0,:] = sol[-1, indx_spectrum[ix], :]
        if abs(loc_spectrum[ix] - shock_location) <= 1.e-2:
            np.savetxt(internal_folder + 'D-Kol/t-final_D-Kol_x=Shock.txt', data_txt_file)
            np.savetxt(dirName + 't-final_D-Kol_x=Shock.txt', data_txt_file)
        else:
            np.savetxt(internal_folder + 'D-Kol/t-final_D-Kol_x=' + str( round(loc_spectrum[ix]*L_ref) ) + 'pc.txt', data_txt_file)
            np.savetxt(dirName + 't-final_D-Kol_x=' + str( round(loc_spectrum[ix]*L_ref) ) + 'pc.txt', data_txt_file)

            
    elif delta_diff == 0.5:
        
        if ix == 0:
            try:
                # Create target Directory
                os.mkdir(internal_folder + 'D-IK/')
                print("Directory", internal_folder + 'D-IK/', "created")
                print("")
            except FileExistsError:
                print("Directory", internal_folder + 'D-IK/', "already exists")
                print("")
        
        data_txt_file = np.zeros( (1, len(p_grid)), dtype=np.float64 )
        data_txt_file[0,:] = sol[-1, indx_spectrum[ix], :]
        if abs(loc_spectrum[ix] - shock_location) <= 1.e-2:
            np.savetxt(internal_folder + 'D-IK/t-final_D-IK_x=Shock.txt', data_txt_file)
            np.savetxt(dirName + 't-final_D-IK_x=Shock.txt', data_txt_file)
        else:
            np.savetxt(internal_folder + 'D-IK/t-final_D-IK_x=' + str( round(loc_spectrum[ix]*L_ref) ) + 'pc.txt', data_txt_file)
            np.savetxt(dirName + 't-final_D-IK_x=' + str( round(loc_spectrum[ix]*L_ref) ) + 'pc.txt', data_txt_file)
            
            
    elif delta_diff == 1.:
        
        if ix == 0:
            try:
                # Create target Directory
                os.mkdir(internal_folder + 'D-Bohm/')
                print("Directory", internal_folder + 'D-Bohm/', "created")
                print("")
            except FileExistsError:
                print("Directory", internal_folder + 'D-Bohm/', "already exists")
                print("")
        
        data_txt_file = np.zeros( (1, len(p_grid)), dtype=np.float64 )
        data_txt_file[0,:] = sol[-1, indx_spectrum[ix], :]
        if abs(loc_spectrum[ix] - shock_location) <= 1.e-2:
            np.savetxt(internal_folder + 'D-Bohm/t-final_D-Bohm_x=Shock.txt', data_txt_file)
            np.savetxt(dirName + 't-final_D-Bohm_x=Shock.txt', data_txt_file)
        else:
            np.savetxt(internal_folder + 'D-Bohm/t-final_D-Bohm_x=' + str( round(loc_spectrum[ix]*L_ref) ) + 'pc.txt', data_txt_file)
            np.savetxt(dirName + 't-final_D-Bohm_x=' + str( round(loc_spectrum[ix]*L_ref) ) + 'pc.txt', data_txt_file)



########################################################################
## inelastic cross-section for proton energy in [GeV]; result in [mb] ##
########################################################################
def sigma_inel(Tp):

    Tp_th = 1.22              # threshold energy for the pi0 production, in [GeV]
    LX = np.log( Tp/Tp_th )
    Threshold = max( 0., 1. - ( Tp_th/Tp )**1.9 )

    if Tp >= Tp_th:
        #return ( 30.7 - 0.96*LX + 0.18*LX**2. ) * Threshold**3.
        return ( 34.3 - 1.88*LX + 0.25*LX**2. ) * Threshold**3.
    else:
        return 0.

    

def sigma_gamma(E_proj, E_gamma):
# result given in [mb / GeV]

    if E_gamma >= E_proj:
        return 0.
    
    proton_mass = m_p
    TeV = 1.e3
    E_p = E_proj
    L = np.log(E_p / TeV)     # defined in pag.9

    x = E_gamma / E_p         # defined in pag.9

    B_gamma = 1.30 + 0.14 * L + 0.011 * L**2              # Eq.(59)
    beta_gamma = 1. / (1.79 + 0.11 * L + 0.008 * L**2)    # Eq.(60)
    k_gamma = 1. / (0.801 + 0.049 * L + 0.014 * L**2)     # Eq.(61)
    x2beta = x**(beta_gamma)

    F_1 = (1. - x2beta) / (1. + k_gamma * x2beta * (1. - x2beta))
    F_2 = 4. * beta_gamma * x2beta / (1. - x2beta)
    F_3 = 4. * k_gamma * beta_gamma * x2beta * (1. - 2. * x2beta)
    F_3 = F_3 / ( 1. + k_gamma * x2beta * (1. - x2beta) )

    F_gamma = B_gamma * np.log(x) / x * F_1**4            # Eq.(58)
    F_gamma = F_gamma * (1. / np.log(x) - F_2 - F_3)

    return sigma_inel(E_p) * F_gamma / E_p



def F_gamma_func(E_proj, x_):

    proton_mass = m_p
    TeV = 1.e3
    E_p = E_proj
    L = np.log(E_p / TeV)     # defined in pag.9

    x = x_

    B_gamma = 1.30 + 0.14 * L + 0.011 * L**2              # Eq.(59)
    beta_gamma = 1. / (1.79 + 0.11 * L + 0.008 * L**2)    # Eq.(60)
    k_gamma = 1. / (0.801 + 0.049 * L + 0.014 * L**2)     # Eq.(61)
    x2beta = x**(beta_gamma)

    F_1 = (1. - x2beta) / (1. + k_gamma * x2beta * (1. - x2beta))
    F_2 = 4. * beta_gamma * x2beta / (1. - x2beta)
    F_3 = 4. * k_gamma * beta_gamma * x2beta * (1. - 2. * x2beta)
    F_3 = F_3 / ( 1. + k_gamma * x2beta * (1. - x2beta) )

    F_gamma = B_gamma * np.log(x) / x * F_1**4            # Eq.(58)
    F_gamma = F_gamma * (1. / np.log(x) - F_2 - F_3)

    return F_gamma



x_Xsec = np.logspace(start=-5., stop=-0., num=100)


plt.figure(figsize=(13,5))
plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.plot(p_grid, [sigma_inel( p_grid[ip] ) for ip in range(len(p_grid))], lw=2., color='blue')
plt.xscale('log')
plt.xlabel('$p_{\mathrm{proj}} \, [\mathrm{GeV}/c]$', fontsize=20)
plt.ylabel('$\\sigma^{pp}_{\mathrm{inel}} \, [\mathrm{mb}]$', fontsize=20)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()


p_show_Xsec = [1.e3, 3.e4, 3.e5, 3.e6]
colors = ['blue', 'red', 'orange', 'green']
for ip in range( len(p_show_Xsec) ):
    plt.loglog(x_Xsec, [p_show_Xsec[ip] * x_Xsec[ix]**2 * sigma_gamma(p_show_Xsec[ip], p_show_Xsec[ip]*x_Xsec[ix]) for ix in range(len(x_Xsec))], color=colors[ip],  lw=2., label='$p_\mathrm{proj} = \,$' + str(p_show_Xsec[ip]/1.e3) + '$\, \mathrm{TeV}$')
    
    
plt.xlabel('$x \equiv E_{\gamma} \\big/ E_{p}$', fontsize=20)
plt.ylabel('$x^2 \cdot \\frac{d \\sigma}{dx}(x, E_\gamma) $', fontsize=20)
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()



d_cluster = 1.4                       # in [kpc]
d_cluster_cm = d_cluster * 1.e3 * conv_pc_cm
radius_gammaray_integration = 55.     # in [pc], integration radius considered by HAWC&Fermi
x_grid_pc = x_grid * L_ref
indx_radius_gammaray_integration = np.argmin( abs(radius_gammaray_integration - x_grid_pc) )
print(f'the integration to compute the gamma-rays is done up to index {indx_radius_gammaray_integration}, i.e.: {x_grid_pc[indx_radius_gammaray_integration]} [pc]')

log_A = np.log(factor_grid[0])
E_gamma = np.logspace(start=0., stop=np.log10( max(p_grid)/10 ), num=250)    # where we want to see the gamma rays


summation = np.zeros( (len(x_grid), len(E_gamma)), dtype=np.float64 )
emissivity_gamma = np.zeros( (len(x_grid), len(E_gamma)), dtype=np.float64 )
emissivity_gamma_rsquared = np.zeros( (len(x_grid), len(E_gamma)), dtype=np.float64 )
Flux_gamma = np.zeros( len(E_gamma), dtype=np.float64 )



n_p_PhSp = np.zeros( (len(x_grid), len(p_array_integral_tot)), dtype=np.float64 )
n_p = np.zeros( (len(x_grid), len(p_array_integral_tot)), dtype=np.float64 )
n_p_PhSp = CR_spectrum * CR_norm
for ix in range(len(x_grid)):
    n_p[ix, :] = [4 * np.pi * p_array_integral_tot[ip]**2 * n_p_PhSp[ix, ip] for ip in range(len(p_array_integral_tot))]

    

plt.figure(figsize=(14, 5.5))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()

for ix in range(len(loc_spectrum)):
    if loc_spectrum[ix] <= shock_location:
        if abs(loc_spectrum[ix] - shock_location) <= 1.e-2:
            plt.loglog(p_array_integral_tot[:], (p_array_integral_tot[:] / p_grid[0])**2. * n_p[indx_spectrum[ix], :], ls='-', lw=2.5, color='blue', label='$r_{\mathrm{shock}} \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')
        else:
            plt.loglog(p_array_integral_tot[:], (p_array_integral_tot[:] / p_grid[0])**2. * n_p[indx_spectrum[ix], :], ls='-', lw=1.5, label='$r \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')

    
plt.ylim(np.max((p_array_integral_tot[:] / p_grid[0])**2 * n_p[:, :]) / 1.e5, np.max((p_array_integral_tot[:]/p_grid[0])**2 * n_p[:, :]) * 1.e1)
#plt.ylim(np.max((p_array_integral_tot[:] / p_grid[0])**2 * n_p[:, :]) / 3.e2, np.max((p_array_integral_tot[:]/p_grid[0])**2 * n_p[:, :]) * 1.5)
plt.title('Upstream energy spectra', fontsize=20, pad=8)
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
plt.ylabel('$E^2 \cdot N(E,T_{\mathrm{run}}) \; [\mathrm{GeV \cdot cm^{-3}}]$', fontsize=20)
plt.text(0.65, 0.9, '$r_{\mathrm{shock}} = \,$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=18, transform = plt.gca().transAxes)
plt.legend(fontsize=16, frameon=False)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

for ix in range(len(loc_spectrum)):
    if loc_spectrum[ix] >= shock_location:
        if abs(loc_spectrum[ix] - shock_location) <= 1.e-2:
            plt.loglog(p_array_integral_tot[:], (p_array_integral_tot[:] / p_grid[0])**2. * n_p[indx_spectrum[ix], :], ls='-', lw=2.5, color='blue', label='$r_{\mathrm{shock}} \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')
        else:
            plt.loglog(p_array_integral_tot[:], (p_array_integral_tot[:] / p_grid[0])**2. * n_p[indx_spectrum[ix], :], ls='-', lw=1.5, label='$r \simeq \,$' + str( round(loc_spectrum[ix]*L_ref) ) + '$\, \mathrm{pc}$')

    

plt.ylim(np.max((p_array_integral_tot[:] / p_grid[0])**2 * n_p[:, :]) / 1.e5, np.max((p_array_integral_tot[:]/p_grid[0])**2 * n_p[:, :]) * 1.e1)
plt.title('Downstream energy spectra', fontsize=20, pad=8)
plt.xlabel('$p \, [\mathrm{GeV}/c]$', fontsize=20)
plt.text(0.65, 0.9, '$r_{\mathrm{shock}} = \,$' + str("{:.2f}".format(shock_location*L_ref)) + '$\, \mathrm{pc}$', fontsize=18, transform = plt.gca().transAxes)
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()

    
start = time.process_time()


# integration along the LoS
###########################
for ig in range( len(E_gamma) ):
    # result in [GeV^{-1} s^{-1} pc^{-3}]
    summation[:, ig] = sum( [p_array_integral_tot[ip] * n_p[:, ip] * sigma_gamma(p_array_integral_tot[ip], E_gamma[ig])*conv_mbarn_cm2 for ip in range( len(p_array_integral_tot) )] )
    emissivity_gamma[:, ig] = c_cm * n_bubble_array[:] * log_A * summation[:, ig] / (conv_cm_pc**3)
    

GammaFlux_array = np.linspace(start=0., stop=x_grid_pc[indx_radius_gammaray_integration], num=100)
integral_over_r = np.zeros( len(GammaFlux_array), dtype=np.float64 )
Flux_gamma = np.zeros( len(E_gamma), dtype=np.float64 )

for ig in range( len(E_gamma) ):
    
    for ir in range( len(GammaFlux_array) ):

        ir_indx_x_grid = np.argmin( abs(GammaFlux_array[ir] - x_grid_pc) )
        n_decades_LoS = round( (L_ref - GammaFlux_array[ir]) / 10. )
        n_PointsPerDecade = 20
        Fy_LoS_array = np.linspace(start=GammaFlux_array[ir]+0.01, stop=L_ref, num=n_decades_LoS * n_PointsPerDecade)
        
        integral_over_r[ir] = 2 * np.trapz( Fy_LoS_array / np.sqrt(Fy_LoS_array**2 - GammaFlux_array[ir]**2) * emissivity_gamma[ir_indx_x_grid, ig], Fy_LoS_array, axis=-1 )
        
    # flux in [GeV^{-1} s^{-1} cm^{-2}]
    Flux_gamma[ig] = np.trapz( integral_over_r * GammaFlux_array, GammaFlux_array, axis=-1 ) / (2 * d_cluster_cm**2)
###########################

    
time_emissivity = time.process_time() - start
print ('It took', time_emissivity, "seconds")

plt.savefig(dirName + 'EnergySpectrum_DifferentLocations_delta=' + str(delta_diff) + '.pdf',format='pdf',bbox_inches='tight', dpi=200)



########################################
## Comparison with the gamma-ray data ##
########################################
from numbers_parser import Document

doc_path = '/Users/ottaviofornieri/PHYSICS_projects/GitProjects/StellarClusters/Data/'
doc = Document(doc_path + 'Fermi+HAWC.numbers')


sheets = doc.sheets
tables = sheets[0].tables
data = tables[0].rows(values_only=True)
df = pd.DataFrame(data[1:], columns=data[0])

print('len of the gamma-ray Fermi+HAWC data:', len(data))
print('')


E_mean_Fermi = np.zeros( 4, dtype=np.float64 )
E_low_Fermi = np.zeros( 4, dtype=np.float64 )
E_up_Fermi = np.zeros( 4, dtype=np.float64 )
E_mean_HAWC = np.zeros( 8, dtype=np.float64 )

Flux_Fermi_mean = np.zeros( 4, dtype=np.float64 )
Flux_HAWC_mean = np.zeros( 8, dtype=np.float64 )
Flux_HAWC_low = np.zeros( 8, dtype=np.float64 )
Flux_HAWC_up = np.zeros( 8, dtype=np.float64 )

E_mean_Fermi[:] = [ data[ie][1] for ie in range(1, 5) ]
E_mean_HAWC[:] = [ data[ie][1] for ie in range(5, len(data)) ]
E_low_Fermi[:] = [ data[ie][2] for ie in range(1, 5) ]
E_up_Fermi[:] = [ data[ie][3] for ie in range(1, 5) ]
Flux_Fermi_mean[:] = [ data[ie][4] for ie in range(1, 5) ]
Flux_HAWC_mean[:] = [ data[ie][4] for ie in range(5, len(data)) ]
Flux_HAWC_low[:] = [ data[ie][5] for ie in range(5, len(data)) ]
Flux_HAWC_up[:] = [ data[ie][6] for ie in range(5, len(data)) ]



plt.figure(figsize=(6.5,4.5))
plot_cosmetics_single()


plt.plot(E_mean_Fermi, Flux_Fermi_mean, 'ro')
plt.plot(E_mean_HAWC, Flux_HAWC_mean, 'bo')
plt.errorbar(E_mean_Fermi, Flux_Fermi_mean, fmt='ro', label='Fermi 4FGL', xerr=[E_mean_Fermi-E_low_Fermi, E_up_Fermi-E_mean_Fermi], yerr=0., ecolor='red')
plt.errorbar(E_mean_HAWC, Flux_HAWC_mean, fmt='bo', label='HAWC', xerr=0., yerr=[Flux_HAWC_mean-Flux_HAWC_low, Flux_HAWC_up-Flux_HAWC_mean], ecolor='blue')


plt.loglog(E_gamma, conv_GeV_TeV * E_gamma**2. * Flux_gamma, lw=2.5, color='blue', label='num $\gamma$-rays')
plt.ylim(1.e-13, 3.e-10)
plt.text(0.05, 0.7, '$T_{\mathrm{run}} = \,$' + str("{:.0f}".format(t_run_Myr)) + '$\, \mathrm{Myr}$', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.05, 0.6, '$R_{\mathrm{TS}} \simeq \,$' + str(round(R_TS)) + '$\, \mathrm{pc}$', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.05, 0.49, '$\\Delta r_{\gamma} \simeq \,$' + str(round(x_grid_pc[indx_radius_gammaray_integration])) + '$\, \mathrm{pc}$', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.55, 0.17, '$\\xi_{\mathrm{CR}} \simeq \,$' + str("{:.0f}".format(xi_CR*100.)) + '\%', fontsize=18, transform = plt.gca().transAxes)
plt.text(0.55, 0.07, '$n_{\mathrm{down}} = \,$' + str("{:.0f}".format(n_bubble)) + '$\, \mathrm{cm^{-3}}$', fontsize=18, transform = plt.gca().transAxes)
plt.xlabel('$E_\gamma \, [\mathrm{GeV}]$', fontsize=20)
plt.ylabel('$E_\gamma^2 \cdot \Phi_\gamma \, [\mathrm{TeV \cdot cm^{-2} \cdot s^{-1}}]$', fontsize=20)
plt.legend(fontsize=16, frameon=False, loc='lower left')

plt.savefig(dirName + 'GammaRay_Spectrum_delta=' + str(delta_diff) + '.pdf',format='pdf',bbox_inches='tight', dpi=200)
