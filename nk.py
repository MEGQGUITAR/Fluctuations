import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# --- Parámetros y funciones de fondo ---
# Define los parámetros y las funciones V3, DV3, D2V3
# que son necesarias para el código.
lambdam = 1.0 # Ejemplo, reemplaza con tu valor
Mp = 1.0 # Ejemplo, reemplaza con tu valor
phi_at_interseccion_6 = 1.0 # Ejemplo, reemplaza con tu valor

def V(phi):
    return lambdam*phi**6
def DV(phi):
    return 6*lambdam*phi**5
def D2V(phi):
    return 30*lambdam*phi**4

# --- Solución de las ecuaciones de movimiento de fondo ---
t0=0
t_target=4813795.41
t_background = np.linspace(t0, t_target, 100000)

def background_eom(t, y):
    phi_bar, phi_bar_dot = y
    dphi_bar_dt = phi_bar_dot
    H = np.sqrt((0.5 * phi_bar_dot**2 + V(phi_bar)) / (3 * Mp**2))
    dphi_bar_dot_dt = -3 * H * phi_bar_dot - DV(phi_bar)
    return [dphi_bar_dt, dphi_bar_dot_dt]

sol_phi_bar = solve_ivp(background_eom, (t_background[0], t_background[-1]),
                        y0=[1.983642, -4.410213],
                        t_eval=t_background, rtol=1e-8, atol=1e-10)

phi_bar_sol = sol_phi_bar.y[0]
phi_bar_dot_sol = sol_phi_bar.y[1]

# --- Interpolación de las soluciones de fondo ---
phi_bar_interp = interp1d(t_background, phi_bar_sol, kind='cubic', fill_value="extrapolate")
phi_bar_dot_interp = interp1d(t_background, phi_bar_dot_sol, kind='cubic', fill_value="extrapolate")
m_end = np.sqrt(D2V(phi_at_interseccion_6))
H_t = np.sqrt((0.5 * phi_bar_dot_sol**2 + V(phi_bar_sol)) / (3 * Mp**2))
a_t = np.exp(np.cumsum(H_t * np.diff(np.concatenate(([t_background[0]], t_background)))))
a_interp = interp1d(t_background, a_t, kind='cubic', fill_value="extrapolate")

# --- Ecuaciones de movimiento de la perturbación (Floquet) ---
def floquet_system(t, y_complex, k_val, phi_bar_interp_func, phi_bar_dot_interp_func, a_interp_func, use_metric_fluctuations):
    re_chi, im_chi, re_chi_dot, im_chi_dot = y_complex
    phi_bar_at_t = phi_bar_interp_func(t)
    phi_bar_dot_at_t = phi_bar_dot_interp_func(t)
    a_at_t = a_interp_func(t)

    # Prevenir divisiones por cero
    if a_at_t < 1e-10:
        a_at_t = 1e-10
    
    m_eff_sq = (k_val / a_at_t)**2
    if use_metric_fluctuations:
        H = np.sqrt((0.5 * phi_bar_dot_at_t**2 + V(phi_bar_at_t)) / (3 * Mp**2))
        m_eff_sq += 3*phi_bar_dot_at_t**2 - ((phi_bar_dot_at_t**4) / (2 * H**2)) + 2*(phi_bar_dot_at_t/H)*DV(phi_bar_at_t) + D2V(phi_bar_at_t)
    else:
        m_eff_sq += D2V(phi_bar_at_t)

    dre_chi_dt = re_chi_dot
    dim_chi_dt = im_chi_dot
    dre_chi_dot_dt = -m_eff_sq * re_chi
    dim_chi_dot_dt = -m_eff_sq * im_chi
    
    return [dre_chi_dt, dim_chi_dt, dre_chi_dot_dt, dim_chi_dot_dt]

# --- Bucle principal y cálculos de n_k ---
k_values = np.logspace(-2, 1, 100)
n_k_simple = np.zeros_like(k_values, dtype=float)
n_k_fluctuations = np.zeros_like(k_values, dtype=float)

t_span_chi = (t_background[0], t_background[-1])
t_eval_chi = t_background

t_initial = t_background[0]
t_final = t_background[-1]
a_initial = max(a_interp(t_initial), 1e-10)
a_final = max(a_interp(t_final), 1e-10)
phi_bar_initial = phi_bar_interp(t_initial)
phi_bar_dot_initial = phi_bar_dot_interp(t_initial)
H_initial = np.sqrt((0.5 * phi_bar_dot_initial**2 + V(phi_bar_initial)) / (3 * Mp**2))
phi_bar_at_final_t = phi_bar_interp(t_final)
phi_bar_dot_at_final_t = phi_bar_dot_interp(t_final)
H_final = np.sqrt((0.5 * phi_bar_dot_at_final_t**2 + V(phi_bar_at_final_t)) / (3 * Mp**2))

print("Calculando n_k con y sin fluctuaciones de la métrica...")

for i, k_val_loop in enumerate(k_values):
    print(f"Calculando {i+1} de {len(k_values)}")
    k_val = k_val_loop * m_end

    # --- Cálculo de n_k sin fluctuaciones (versión simple) ---
    omega_k_initial_sq_simple = k_val**2 / a_initial**2 + D2V(phi_bar_initial)
    omega_k_initial_simple = np.sqrt(max(omega_k_initial_sq_simple, 1e-10))
    y0_simple = [1.0 / np.sqrt(2.0 * omega_k_initial_simple), 0.0, 0.0, -np.sqrt(omega_k_initial_simple / 2.0)]

    sol_simple = solve_ivp(floquet_system, t_span_chi, y0=y0_simple,
                           args=(k_val, phi_bar_interp, phi_bar_dot_interp, a_interp, False), t_eval=t_eval_chi,
                           rtol=1e-8, atol=1e-10)

    chi_final_simple = sol_simple.y[0, -1] + 1j * sol_simple.y[1, -1]
    chi_prime_final_simple = sol_simple.y[2, -1] + 1j * sol_simple.y[3, -1]

    omega_k_final_sq_simple = k_val**2 / a_final**2 + D2V(phi_bar_at_final_t)
    omega_k_final_simple = np.sqrt(max(omega_k_final_sq_simple, 1e-10))
    term_for_beta_simple = omega_k_final_simple * chi_final_simple - 1j * chi_prime_final_simple
    n_k_simple[i] = np.abs(term_for_beta_simple)**2 / (2 * omega_k_final_simple)

    # --- Cálculo de n_k con fluctuaciones de la métrica ---
    omega_k_initial_sq_fluct = k_val**2 / a_initial**2 + 3*phi_bar_dot_initial**2 - ((phi_bar_dot_initial**4) / (2 * H_initial**2)) + 2*(phi_bar_dot_initial/H_initial)*DV(phi_bar_initial) + D2V(phi_bar_initial)
    omega_k_initial_fluct = np.sqrt(max(omega_k_initial_sq_fluct, 1e-10))
    y0_fluct = [1.0 / np.sqrt(2.0 * omega_k_initial_fluct), 0.0, 0.0, -np.sqrt(omega_k_initial_fluct / 2.0)]
    
    sol_fluct = solve_ivp(floquet_system, t_span_chi, y0=y0_fluct,
                          args=(k_val, phi_bar_interp, phi_bar_dot_interp, a_interp, True), t_eval=t_eval_chi,
                          rtol=1e-8, atol=1e-10)

    chi_final_fluct = sol_fluct.y[0, -1] + 1j * sol_fluct.y[1, -1]
    chi_prime_final_fluct = sol_fluct.y[2, -1] + 1j * sol_fluct.y[3, -1]

    omega_k_final_sq_fluct = k_val**2 / a_final**2 + 3*phi_bar_dot_at_final_t**2-((phi_bar_dot_at_final_t**4)/(2*H_final**2))+2*(phi_bar_dot_at_final_t/H_final)*DV(phi_bar_at_final_t)+D2V(phi_bar_at_final_t)
    omega_k_final_fluct = np.sqrt(max(omega_k_final_sq_fluct, 1e-10))

    term_for_beta_fluct = omega_k_final_fluct * chi_final_fluct - 1j * chi_prime_final_fluct
    n_k_fluctuations[i] = np.abs(term_for_beta_fluct)**2 / (2 * omega_k_final_fluct)

print("Cálculo de n_k finalizado. Guardando resultados...")

# --- Generación del archivo de texto ---
data_to_save = np.column_stack((k_values * m_end, n_k_simple, n_k_fluctuations))
header = "k\tn_k (sin fluctuaciones)\tn_k (con fluctuaciones)"
np.savetxt("resultados_nk.txt", data_to_save, fmt='%.18e', delimiter='\t', header=header)

print("Archivo 'resultados_nk.txt' generado exitosamente.")
