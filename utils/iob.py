import numpy as np

def ExponentialIOB(time: float, time_peak: float, time_duration: float) -> float:
  """
  Function to calculate insulin on board with exponential decay scheme.
  Parameters:
    - time: time after bolus delivery
    - time_peak: characteristics of particular insulin - time of peak action
    - time_duration: characteristics of particular insulin - duration of insulin action
  Returns: current insulin on board

  More information about IoB: 
  http://guidelines.diabetes.ca/cdacpg_resources/Ch12_Table1_Types_of_Insulin_updated_Aug_5.pdf
  """

  if time > time_duration:
    return 0

  peak_duration_ratio = time_peak/time_duration
  tau_coef = time_peak*(1 - peak_duration_ratio)/(1 - 2*peak_duration_ratio)
  alpha_coef = 2*tau_coef/time_duration
  s_coef = 1/(1 - alpha_coef + (1 + alpha_coef)*np.exp(-time_duration/tau_coef))
  insulin_on_board = 1 - s_coef*(1 - alpha_coef)/
                     *((time**2/(tau_coef*time_duration * (1 - alpha_coef)) - time/tau_coef - 1)/
                     *np.exp(-time/tau_coef) + 1)
  return insulin_on_board

def IoB(times: np.array, time_peak: float, time_duration: float) -> np.array:
  pass
