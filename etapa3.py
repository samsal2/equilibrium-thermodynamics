
###############################################################################
# Librerias utilizadas
###############################################################################


import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# Declaración de metodos
###############################################################################


def nrtl_gamma_sistema_binario_substancia_1(x1, t12, t21, G12, G21):
  """
  gamma 1 descrita por la ecuación 5-62 del seader

  gamma1 = e^[x2^2[(t21G21^2) / (x1 + x2G21)^2 + (t12G12) / (x2 + x1G12)^2]]

  Parametros
  ----------
  x1 : float
    Fracción molar de la substancia 1
  t12 : float
    Parametro t de NRTL 12
  t21 : float
    Parametro t de NRTL 21
  G12 : float
    Parametro G de NRTL 12
  G21 : float
    Parametro G de nRTL 21

  Devuelve
  --------
  float
    gamma 1
  """
  x2 = 1 - x1

  # gamma1 = e^[x2^2[(t21G21^2) / (x1 + x2G21)^2 + (t12G12) / (x2 + x1G12)^2]]
  # a = (t21G21^2) / (x1 + x2G21)^2
  # b = (t12G12) / (x2 + x1G12)^2
  # gamma1 = e^[x2^2[a + b]]

  # Precalculando valores
  a = (t21 * G21 * G21) / ((x1 + x2 * G21) * (x1 + x2 * G21))
  b = (t12 * G12) / ((x2 + x1 * G12) * (x2 + x1 * G12))

  return np.exp(x2 * x2 * (a + b))


def nrtl_gamma_sistema_binario_substancia_2(x2, t12, t21, G12, G21):
  """
  gamma 2 descrita por la ecuación 5-62 del seader

  gamma2 = e^[x1^2[(t12G12^2) / (x2 + x1G12)^2 + (t21G21) / (x1 + x2G21)^2]]

  Parametros
  ----------
  x2 : float
    Fracción molar de la substancia 2
  t12 : float
    Parametro t de NRTL 12
  t21 : float
    Parametro t de NRTL 21
  G12 : float
    Parametro G de NRTL 12
  G21 : float
    Parametro G de nRTL 21

  Devuelve
  --------
  float
    gamma 2
  """
  x1 = 1 - x2

  # gamma1 = e^[x2^2[(t21G21^2) / (x1 + x2G21)^2 + (t12G12) / (x2 + x1G12)^2]]
  # a = (t21G21^2) / (x1 + x2G21)^2
  # b = (t12G12) / (x2 + x1G12)^2
  # gamma1 = e^[x2^2[a + b]]

  # Precalculando valores
  a = (t12 * G12 * G12) / ((x2 + x1 * G12) * (x2 + x1 * G12))
  b = (t21 * G21) / ((x1 + x2 * G21) * (x1 + x2 * G21))

  return np.exp(x1 * x1 * (a + b))


def nrtl_tau(delta_g, R, T):
  """
  τ_ij o τ_ji descrita por la ecuación 5-60 o 5-61 del seader

  si delta_g = (g_ij - g_jj)
  τ_ij = (g_ij - g_jj) / (RT)

  si delta_g = (g_ji - g_jj)
  τ_ji = (g_ji - g_ii) / (RT)

  Parametros
  ----------
  delta_g : float
    La diferencia entre g_ij - g_jj
  R : float
    Constante universal de los gases
  T : float
    Temperatura

  Devuelve
  --------
  float
    τ_ij
  """

  return delta_g / (R * T)


def nrtl_G(alfa_ji, tau_ji):
  """
  G_ij expresado por la ecuación 6-59 del seader

  G_ij = e^(-α_jiτ_ji)

  Parametros:
  -----------
  alfa_ji : float
    α_ji
  tau_ji : float
    τ_ji

  Devuelve:
  ---------
  float
    G_ij
  """

  return np.exp(-alfa_ji * tau_ji)


def nrtl_delta_g(Aij, Bij, Cij, T, Tc):
  """
  Ecuación para sacar g_ij - g_jj

  g_ij - g_jj = Aij + Bij(Tc - T) + Cij(Tc - T)^@

  Parametros
  ----------
  Aij : float
    Primer coeficiente del polinomio
  Bij : float
    Segundo coeficiente del polinomio
  Cij : float
    Tercer coeficiente del polinomio
  T : float
    Temperatura
  Tc : float
    Temperatura critica

  Devuelve:
    g_ij - g_jj
  """

  t = Tc - T
  return Aij + Bij * t + Cij * t * t


def encontrar_x_1_con_parametros_para_delta_g(Aij, Bij, Cij, T, R):

  # Por ELL gamma_1 * x_1 = gamma_2 * x_2

  pass


def encontrar_parametros_para_nrtl_delta_g(T_experimental, x_experimental):
  pass


def 
