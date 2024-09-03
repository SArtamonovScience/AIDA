def hovorka_parameters(BW):
    """
    PATIENT PARAMETERS
    BW - body weight in kilos, BW = params.BW
    Original source: https://github.com/jonasnm/gym/blob/master/gym/envs/diabetes/hovorka_model.py
    """

    # Patient-dependent parameters:
    V_I = 0.12*BW              # Insulin volume [L]
    V_G = 0.16*BW              # Glucose volume [L]
    F_01 = 0.0097*BW           # Non-insulin-dependent glucose flux [mmol/min]
    EGP_0 = 0.0161*BW          # EGP extrapolated to zero insulin concentration [mmol/min]

    # Patient-independent(?) parameters:
    S_IT = 51.2e-4             # Insulin sensitivity of distribution/transport [L/min*mU]
    S_ID = 8.2e-4              # Insulin sensitivity of disposal [L/min*mU]
    S_IE = 520e-4              # Insluin sensitivity of EGP [L/mU]

    tau_G = 40                 # Time-to-maximum CHO absorption [min]
    tau_I = 55                 # Time-to-maximum of absorption of s.c. injected short-acting insulin [min]

    A_G = 0.8                  # CHO bioavailability [1]
    k_12 = 0.066               # Transfer rate [min]

    k_a1 = 0.006               # Deactivation rate of insulin on distribution/transport [1/min]
    k_b1 = S_IT*k_a1           # Activation rate of insulin on distribution/transport
    k_a2 = 0.06                # Deactivation rate of insulin on dsiposal [1/min]
    k_b2 = S_ID*k_a2           # Activation rate of insulin on disposal
    k_a3 = 0.03                # Deactivation rate of insulin on EGP [1/min]
    k_b3 = S_IE*k_a3           # Activation rate of insulin on EGP

    k_e = 0.138                # Insulin elimination from Plasma [1/min]

    # Summary of the patient's values:
    P = [tau_G, tau_I, A_G, k_12, k_a1, k_b1, k_a2, k_b2, k_a3, k_b3, k_e, V_I, V_G, F_01, EGP_0]
    return P


def hovorka_model(x: list, t: float, u: float, D: float, P: list) -> list:
    """
    One step of ODE Hovorka solution
    Parameters:
        - x: list of current parameters
        - t: time
        - u: speed of insulin delivery, un/min
        - D: speed of carbonhydrates delivery, mmol/min
        - P: list of patients parameters
    Returns:
        List of increments of current parameters (dx)
    """
    D1, D2, S1, S2, Q1, Q2, I, x1, x2, x3, C = x # Unpack current ODE parameters
    tau_G, tau_I, A_G, k_12, k_a1, k_b1, k_a2, k_b2, k_a3, k_b3, k_e, V_I, V_G, F_01, EGP_0 = P # Unpack patient constant parameters

    U_G = D2/tau_G             # Normalized speed of carbonhydrate delivery[mmol/min]
    U_I = S2/tau_I             # Скорость поглощения инсулина [mU/min]

    G = Q1/V_G                 # Концентрация глюкозы [mmol/L]

    if G >= 4.5:
        F_01c = F_01           # Потребление глюкозы ЦНС [mmol/min]
    else:
        F_01c = F_01 * G / 4.5

    if G >= 9:
        F_R = 0.003 * (G - 9) * V_G  # Почечная экскреция глюкозы [mmol/min]
    else:
        F_R = 0

    # ODE
    dD1 = A_G * D - D1 / tau_G
    dD2 = D1 / tau_G - U_G
    dS1 = u - S1 / tau_I
    dS2 = S1 / tau_I - U_I
    dQ1 = -(F_01c + F_R) - x1 * Q1 + k_12 * Q2 + U_G + EGP_0 * (1 - x3)
    dQ2 = x1 * Q1 - (k_12 + x2) * Q2
    dI = U_I / V_I - k_e * I
    dx1 = k_b1 * I - k_a1 * x1
    dx2 = k_b2 * I - k_a2 * x2
    dx3 = k_b3 * I - k_a3 * x3
    ka_int = 0.073
    dC = ka_int * (G - C)

    return [dD1, dD2, dS1, dS2, dQ1, dQ2, dI, dx1, dx2, dx3, dC]

def simulate_glucose_concentration(time, BW, u, D, initial_bg):
    """Симуляция концентрации глюкозы через заданное время."""
    # Получаем параметры модели для заданного веса тела
    P = hovorka_parameters(BW)
    
    # Начальные условия (все нули)
    x0 = np.zeros(11)
    x0[4] = initial_bg*P[12]
    
    # Время для решения ОДУ от 0 до заданного времени
    t = np.linspace(0, time, num=100)
    
    # Решаем ОДУ
    solution = odeint(hovorka_model, x0, t, args=(u, D, P))
    
    # Получаем конечное значение концентрации глюкозы
    final_glucose_concentration = solution[-1, 4] / P[12]  # Q1 / V_G
    
    return final_glucose_concentration, solution[:, 4] / P[12]
