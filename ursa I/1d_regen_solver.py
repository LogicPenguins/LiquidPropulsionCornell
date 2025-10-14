import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Not really useful for Qc - Qg comparisons (inaccurate - q'' fluctions). Use -> channel design
def calc_total_surface_area(
    L_c, D_c,               # Chamber len and diam [m]
    theta_c_deg, D_t,       # Converging half angle [deg] and throat diam [m]
    L_t,                    # Throat section len [m]
    theta_d_deg, D_e        # Diverging half angle [deg], exit diam [m]
):
    theta_c = np.radians(theta_c_deg)
    theta_d = np.radians(theta_d_deg)
    
    r_c = D_c / 2
    r_t = D_t / 2
    r_e = D_e / 2
    
    A_chamber = np.pi * D_c * L_c
    
    L_conv = (r_c - r_t) / np.tan(theta_c)
    A_conv = np.pi * (r_c + r_t) * np.sqrt((r_c - r_t)**2 + L_conv**2)
    
    A_throat = np.pi * D_t * L_t
    
    L_div = (r_e - r_t) / np.tan(theta_d)
    A_div = np.pi * (r_e + r_t) * np.sqrt((r_e - r_t)**2 + L_div**2)
    
    return A_chamber + A_conv + A_throat + A_div

def validate_total_heat_budget(
    q_array,
    D_inner_array,
    dx_array,
    mdot_cool,
    cp_cool,
    T_cool_in,
    T_cool_crit,
):
    A_segments = np.pi * D_inner_array * dx_array
    Q_gas_total = np.sum(q_array * A_segments)
    Q_cool_total = mdot_cool * cp_cool * (T_cool_crit - T_cool_in)
    
    margin = Q_cool_total - Q_gas_total
    check_pass = Q_gas_total <= Q_cool_total
    
    return check_pass

def area_mach_relation(M, gamma, area_ratio):
    left = (1 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2))**((gamma + 1)/(2 * (gamma - 1)))
    return left - area_ratio
    
def solve_mach(area_ratio, gamma, supersonic):
    if supersonic: 
        M_guess = 2.0 # just a guess for numeric solver
    else:
        M_guess = 0.2 # another guess for numeric solver
    
    M_solution = fsolve(area_mach_relation, M_guess, args=(gamma, area_ratio))
    return M_solution[0]

# Inputs can be regular python lists
def get_combustion_viscosity(mole_fractions, viscosities, molar_masses):
    x = np.array(mole_fractions)
    mu = np.array(viscosities)
    M = np.array(molar_masses)
    
    n = len(x)
    phi = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                phi[i, j] = 1
            else:
                numerator = (1 + np.sqrt(mu[i] / mu[j]) * (M[j] / M[i])**0.25)**2
                denominator = np.sqrt(8 * (1 + M[i] / M[j]))
                phi[i, j] = numerator / denominator

    mu_eff = np.zeros(n)
    for i in range(n):
        denominator_sum = sum(x[j] * phi[i, j] for j in range(n))
        mu_eff[i] = mu[i] / denominator_sum

    mu_mix = sum(x[i] * mu_eff[i] for i in range(n))
    return mu_mix
   
# run regen solver distinctly for chamber, converging, throat, and diverging section
def regen_solver(
    x,              # axial positions array [m]
    D_inner,        # inner diameter along part [m]
    t_wall,         # wall thickness [m]
    R_throat,        # avg radius of throat curvature [m]
    A_throat,       # throat area [m^2]
    D_throat,       # throat diameter [m]
    A_local,        # local cross-sectional area [m^2]
    Pc, Tc,         # chamber pressure [Pa] and chamber temp [K]
    gamma, mu_g, cp_g, pr_g, # local gas properties (gamma, dynamic visc., cp constant press., prandtl's)
    c_star,         # characteristic velocity [m/s]
    Twg_guess,      # initial wall temp guess [K]
    mdot_cool,      # coolant mdot [kg/s]
    T_cool_in,      # coolant inlet temp [K]
    cp_cool, mu_cool, k_cool, pr_cool, rho_cool, # coolant props
    Dh_channel,     # local hydraulic diam of cooalnt channels [m]
    A_channel,      # local cross sectional flow area of coolant channels [m^2]
    k_wall,         # thermal conductivity wall material [W/m-K]
    supersonic,     # boolean val for mach number compressible flow correlation
    T_cool_crit     # known critical coolant temp (acount for pressure) [K]
):
    N = len(x)
    dx = np.diff(x)
    dx = np.append(dx, dx[-1]) # helps w/ keeping simple loop logic w/ consistent length, assume same len [-1]
    
    T_aw = np.zeros(N)
    h_g = np.zeros(N)
    q = np.zeros(N)
    T_wg = np.zeros(N)
    T_wc = np.zeros(N)
    T_cool = np.zeros(N)
    h_c = np.zeros(N)
    M_local = np.zeros(N)
    q_cool_cap = np.zeros(N)
    T_cool_safe = np.ones(N, dtype=bool)
    
    T_cool[0] = T_cool_in
    
    for i in range(N):  
        area_ratio = A_local[i] / A_throat
        M = solve_mach(area_ratio, gamma, supersonic)
        M_local[i] = M

        r = pr_g**(1/3)
        T_aw[i] = Tc * (1 + r * (gamma - 1)/2 * M**2)
        
        T_wg[i] = Twg_guess

        bartz_const = 0.026 / D_throat**0.2
        sigma = 1 / ((0.5 * (T_wg[i] / Tc) * (1 + ((gamma - 1) / 2) * M_local[i]**2)
                 + 0.5)**0.68 * (1 + ((gamma - 1) / 2) * M_local[i]**2)**0.12)
        h_g[i] = (
            bartz_const *
            ((mu_g**0.2 * cp_g) / pr_g**0.6) *
            (((Pc) / c_star)**0.8) *
            ((D_throat / R_throat)**0.1) *
            ((A_throat / A_local[i])**0.9) *
            sigma
        )

                 
        q[i] = h_g[i] * (T_aw[i] - T_wg[i])
        T_wc[i] = T_wg[i] - ((q[i] * t_wall) / k_wall)

        Re_cool = rho_cool * mdot_cool / A_channel / mu_cool * Dh_channel
        Nu = 0.038 * (Re_cool**0.8) * (pr_cool**0.4)
        h_c[i] = Nu * k_cool / Dh_channel
        
        q_cool_cap[i] = h_c[i] * (T_wg[i] - T_cool[i]) # heat flux capacity, diff from Qc
        
        if i < N - 1:
            A_segment = np.pi * D_inner[i] * dx[i]
            Q_in = q[i] * A_segment
            
            delta_T_c = Q_in / (mdot_cool * cp_cool)
            T_cool[i + 1] = T_cool[i] + delta_T_c
            T_cool_safe[i + 1] = T_cool[i + 1] < T_cool_crit
    
        
    df = pd.DataFrame({
        "x [m]": x,
        "dx [m]": dx,
        "Local Mach": M_local,
        "T_aw [K]": T_aw,
        "T_wg [K]": T_wg,
        "T_wc [K]": T_wc,
        "T_cool [K]": T_cool,
        "q_gas [W/m^2]": q,
        "q_cool_cap [W/m^2]": q_cool_cap,
        "h_g [W/m^2-K]": h_g,
        "h_c [W/m^2-K]": h_c,
        "Coolant Safe?": T_cool_safe
    })
    
    return df

# ---------- General Values ---------- #
t_wall = 0.0006
R_throat = 0.00635
A_throat = 0.00012595
D_throat = 0.01266
c_star = 1357
Pc = 1723689
Tc = 1800

mole_frac = [0.628, 0.251, 0.071, 0.039, 0.011]
mu_species = [(4.5*10**-5), (4*10**-5), (9*10**-5), (6*10**-5), (1.2*10**-5)]
molar_mass = [28, 28, 18, 44, 2]
mu_g = get_combustion_viscosity(mole_frac, mu_species, molar_mass)

k_g = 0.18

mdot_cool = 0.053
cp_cool = 2570
mu_cool = (1.074 * 10**-3)
k_cool = 0.166
pr_cool = (mu_cool * cp_cool) / k_cool
rho_cool = 820
Dh_channel = 0.0008
A_channel = 7.2 * (10**-7)
k_wall = 52
T_cool_crit = 484

# ---------- DIVERGING ---------- #
div_x = np.linspace(0, 0.021, 5)
div_d_inner = np.linspace(0.024, 0.0127, 5)
div_A_inner = (np.pi / 4) * div_d_inner**2
div_gamma = 1.2424
div_cp_g = 2.5074
div_pr_g = (div_cp_g * mu_g) / k_g
div_Twg_guess = 750
div_T_cool_in = 233 
div_supersonic = True

# ---------- THROAT ---------- #

thr_x = np.linspace(0.021, 0.024, 2)
thr_d_inner = np.repeat(0.0127, 2)
thr_A_inner = (np.pi / 4) * thr_d_inner**2
thr_gamma = 1.2808
thr_cp_g = 1.9474
thr_pr_g = (thr_cp_g * mu_g) / k_g
thr_Twg_guess = 1000
thr_supersonic = True

# ---------- CONVERGING ---------- #

conv_x = np.linspace(0.024, 0.056, 5)
conv_d_inner = np.linspace(0.0127, 0.02784, 5)
conv_A_inner = (np.pi / 4) * conv_d_inner**2
conv_gamma = 1.2757
conv_cp_g = 1979
conv_pr_g = (conv_cp_g * mu_g) / k_g
conv_Twg_guess = 850
conv_supersonic = False

# ---------- CHAMBER ---------- #
cham_x = np.linspace(0.056, 0.094, 10)
cham_d_inner = np.repeat(0.028, 10)
cham_A_inner = np.repeat(0.000629, 10)
cham_gamma = 1.2757
cham_cp_g = 1979
cham_pr_g = (cham_cp_g * mu_g) / k_g
cham_Twg_guess = 700
cham_supersonic = False

# DIVERGING RUN
df_div = regen_solver(
    x=div_x,
    D_inner=div_d_inner,
    t_wall=t_wall,
    R_throat=R_throat,
    A_throat=A_throat,
    D_throat=D_throat,
    A_local=div_A_inner,
    Pc=Pc,
    Tc=Tc,
    gamma=div_gamma,
    mu_g=mu_g,
    cp_g=div_cp_g,
    pr_g=div_pr_g,
    c_star=c_star,
    Twg_guess=div_Twg_guess,
    mdot_cool=mdot_cool,
    T_cool_in=div_T_cool_in,
    cp_cool=cp_cool,
    mu_cool=mu_cool,
    k_cool=k_cool,
    pr_cool=pr_cool,
    rho_cool=rho_cool,
    Dh_channel=Dh_channel,
    A_channel=A_channel,
    k_wall=k_wall,
    supersonic=div_supersonic,
    T_cool_crit=T_cool_crit
)

# Use prev coolant temp for inlet coolant temp to throat
thr_T_cool_in = df_div["T_cool [K]"].iloc[-1]

# THROAT RUN
df_thr = regen_solver(
    x=thr_x,
    D_inner=thr_d_inner,
    t_wall=t_wall,
    R_throat=R_throat,
    A_throat=A_throat,
    D_throat=D_throat,
    A_local=thr_A_inner,
    Pc=Pc,
    Tc=Tc,
    gamma=thr_gamma,
    mu_g=mu_g,
    cp_g=thr_cp_g,
    pr_g=thr_pr_g,
    c_star=c_star,
    Twg_guess=thr_Twg_guess,
    mdot_cool=mdot_cool,
    T_cool_in=thr_T_cool_in,
    cp_cool=cp_cool,
    mu_cool=mu_cool,
    k_cool=k_cool,
    pr_cool=pr_cool,
    rho_cool=rho_cool,
    Dh_channel=Dh_channel,
    A_channel=A_channel,
    k_wall=k_wall,
    supersonic=thr_supersonic,
    T_cool_crit=T_cool_crit
)

# Use prev coolant temp for inlet coolant temp to converging
conv_T_cool_in = df_thr["T_cool [K]"].iloc[-1]

# CONVERGENT RUN
df_conv = regen_solver(
    x=conv_x,
    D_inner=conv_d_inner,
    t_wall=t_wall,
    R_throat=R_throat,
    A_throat=A_throat,
    D_throat=D_throat,
    A_local=conv_A_inner,
    Pc=Pc,
    Tc=Tc,
    gamma=conv_gamma,
    mu_g=mu_g,
    cp_g=conv_cp_g,
    pr_g=conv_pr_g,
    c_star=c_star,
    Twg_guess=conv_Twg_guess,
    mdot_cool=mdot_cool,
    T_cool_in=conv_T_cool_in,
    cp_cool=cp_cool,
    mu_cool=mu_cool,
    k_cool=k_cool,
    pr_cool=pr_cool,
    rho_cool=rho_cool,
    Dh_channel=Dh_channel,
    A_channel=A_channel,
    k_wall=k_wall,
    supersonic=conv_supersonic,
    T_cool_crit=T_cool_crit
)

# Use prev coolant temp for inlet coolant temp to chamber
cham_T_cool_in = df_conv["T_cool [K]"].iloc[-1]

# CHAMBER RUN
df_cham = regen_solver(
    x=cham_x,
    D_inner=cham_d_inner,
    t_wall=t_wall,
    R_throat=R_throat,
    A_throat=A_throat,
    D_throat=D_throat,
    A_local=cham_A_inner,
    Pc=Pc,
    Tc=Tc,
    gamma=cham_gamma,
    mu_g=mu_g,
    cp_g=cham_cp_g,
    pr_g=cham_pr_g,
    c_star=c_star,
    Twg_guess=cham_Twg_guess,
    mdot_cool=mdot_cool,
    T_cool_in=cham_T_cool_in,
    cp_cool=cp_cool,
    mu_cool=mu_cool,
    k_cool=k_cool,
    pr_cool=pr_cool,
    rho_cool=rho_cool,
    Dh_channel=Dh_channel,
    A_channel=A_channel,
    k_wall=k_wall,
    supersonic=cham_supersonic,
    T_cool_crit=T_cool_crit
)


df_full = pd.concat([df_div, df_thr, df_conv, df_cham], ignore_index=True)

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df_full["x [m]"], df_full["T_cool [K]"], label="Coolant Temperature", color="tab:blue", linewidth=2)
ax1.set_xlabel("Axial Position x [m]")
ax1.set_ylabel("T_cool [K]", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")

ax1.invert_xaxis()

ax2 = ax1.twinx()
ax2.plot(df_full["x [m]"], df_full["q_gas [W/m^2]"], label="Heat Flux (q'')", color="tab:red", linewidth=2)
ax2.set_ylabel("q_gas [W/m²]", color="tab:red")
ax2.tick_params(axis='y', labelcolor="tab:red")

for x_i, safe in zip(df_full["x [m]"], df_full["Coolant Safe?"]):
    color = "green" if safe else "red"
    ax1.axvline(x=x_i, color=color, linestyle="--", alpha=0.2)

fig.tight_layout()
plt.title("Coolant Temperature and Heat Flux vs Axial Position (Chamber -> Exit)")
plt.show()
