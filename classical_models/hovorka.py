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
