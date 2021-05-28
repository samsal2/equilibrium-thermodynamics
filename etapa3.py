##############################################################################
# Librerias utilizadas
###############################################################################


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

###############################################################################
# Declaración de globales 
###############################################################################

# Paramertos de interacción para la equación 5 del
global_parametro_interaccion_A12 = 4.6120810e2
global_parametro_interaccion_B12 = 5.275480e1
global_parametro_interaccion_C12 = 1.718153e-2
global_parametro_interaccion_A21 = 6.523878e3
global_parametro_interaccion_B21 = 6.035228e1
global_parametro_interaccion_C21 = -1.881921
global_alfa_12 = 0.2
global_r_j_molk = 8.134


global_t_experimental = [334.24, 
                         345.48,
                         349.16, 
                         350.52,
                         350.44,
                         349.70,
                         346.09,
                         336.01,
                         327.71,
                         299.71]

global_x_experimental = [0.0991, 
                         0.1669,
                         0.2374, 
                         0.3013,
                         0.4026,
                         0.4984,
                         0.5964,
                         0.7245,
                         0.7640,
                         0.8919]
# Literatura
global_tc_para_mezcla = 349.52

###############################################################################

def rmsep(v_calculado, v_experimental):
  assert len(v_calculado) == len(v_experimental)

  n = len(v_calculado)
 
  s = 0
  for v_calc, v_exp in zip(v_calculado, v_experimental):
    diff = v_calc - v_exp
    s += diff * diff

  return np.sqrt(s / n)

def indice_del_maximo(l):
  j = 0
  v_max = 0
  for i, v_actual in enumerate(l):
    if v_actual > v_max:
      j = i
      v_max = v_actual

  return j


def nrtl_lny_sistema_binario_substancia_1(x, tau, g):
  """
  gamma 1 descrita por la ecuación 5-62 del seader

  gamma1 = e^[x2^2[(t21G21^2) / (x1 + x2G21)^2 + (t12G12) / (x2 + x1G12)^2]]

  Parametros
  ----------
  x : list[float] 
    Lista con los valores de x1 y x2
  t : list[float]
    Lista con los valores de t12 y t21 de NRTL
  g : float
    Lista con los parametros G12 y G21 de NRTL

  Devuelve
  --------
  float
    gamma 1
  """
  x1, x2 = x
  t12, t21 = tau
  g12, g21 = g 


  # Precalculando valores
  a = (t21 * g21 * g21) / ((x1 + x2 * g21) * (x1 + x2 * g21))
  b = (t12 * g12) / ((x2 + x1 * g12) * (x2 + x1 * g12))

  return x2 * x2 * (a + b)


def nrtl_lny_sistema_binario_substancia_2(x, tau, g):
  """
  Parametros
  ----------
  x : list[float] 
    Lista con los valores de x1 y x2
  tau : list[float]
    Lista con los valores de t12 y t21 de NRTL
  g : float
    Lista con los parametros G12 y G21 de NRTL

  Devuelve
  --------
  float
    gamma 2
  """

  x1, x2 = x
  t12, t21 = tau
  g12, g21 = g 

  # Precalculando valores
  a = (t12 * g12 * g12) / ((x2 + x1 * g12) * (x2 + x1 * g12))
  b = (t21 * g21) / ((x1 + x2 * g21) * (x1 + x2 * g21))

  return x1 * x1 * (a + b)



def nrtl_tau(delta_g, r, t):
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
  r : float
    Constante universal de los gases
  t : float
    Temperatura

  Devuelve
  --------
  float
    τ_ij
  """

  return delta_g / (r * t)


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


def nrtl_delta_g(aij, bij, cij, t, tc):
  """
  Ecuación para sacar g_ij - g_jj

  g_ij - g_jj = Aij + Bij(Tc - T) + Cij(Tc - T)^2

  Parametros
  ----------
  aij : float
    Primer coeficiente del polinomio
  bij : float
    Segundo coeficiente del polinomio
  cij : float
    Tercer coeficiente del polinomio
  t : float
    Temperatura
  tc : float
    Temperatura critica

  Devuelve:
    g_ij - g_jj
  """

  t_diff = tc - t
  return aij + bij * t_diff + cij * t_diff * t_diff


def mezcla_calcular_nrtl_delta_g_12(t):
  """
  t (˚K)
  """

  a = global_parametro_interaccion_A12
  b = global_parametro_interaccion_B12
  c = global_parametro_interaccion_C12
  tc = global_tc_para_mezcla
  return nrtl_delta_g(a, b, c, t, tc)


def mezcla_calcular_nrtl_delta_g_21(t):
  """
  t (˚K)
  """

  a = global_parametro_interaccion_A21
  b = global_parametro_interaccion_B21
  c = global_parametro_interaccion_C21
  tc = global_tc_para_mezcla
  return nrtl_delta_g(a, b, c, t, tc)


def mezcla_calcular_funciones_de_t(t):
  delta_g_12 = mezcla_calcular_nrtl_delta_g_12(t)
  delta_g_21 = mezcla_calcular_nrtl_delta_g_21(t)
  tau12 = nrtl_tau(delta_g_12, global_r_j_molk, t)
  tau21 = nrtl_tau(delta_g_21, global_r_j_molk, t)
  G12 = nrtl_G(global_alfa_12, tau12)
  G21 = nrtl_G(global_alfa_12, tau21)
  return [tau12, tau21], [G12, G21]


def funcion_objetivo_1(x1alfa, x1beta, t):
  tau, G = mezcla_calcular_funciones_de_t(t)
  alfa = nrtl_lny_sistema_binario_substancia_1([x1alfa, 1 - x1alfa], tau, G)
  beta = nrtl_lny_sistema_binario_substancia_1([x1beta, 1 - x1beta], tau, G)

  return alfa - beta - np.log(x1beta/x1alfa)


def funcion_objetivo_2(x1alfa, x1beta, t):
  tau, G = mezcla_calcular_funciones_de_t(t)
  alfa = nrtl_lny_sistema_binario_substancia_2([x1alfa, 1 - x1alfa], tau, G)
  beta = nrtl_lny_sistema_binario_substancia_2([x1beta, 1 - x1beta], tau, G)
  
  return alfa - beta - np.log((1 - x1beta) / (1 - x1alfa))


def fsolve_encontrar_x1_alfa_y_x1_beta(t):
  def funcion_objetivo(x):
    obj1 = funcion_objetivo_1(x[0], x[1], t)
    obj2 = funcion_objetivo_2(x[0], x[1], t)
    return obj1, obj2

  return fsolve(funcion_objetivo, [0.0503, 0.8224]) 



t = np.linspace(global_tc_para_mezcla * 1.001, 300, 64)

x_alfa = []
x_beta = []

for t_actual in t:
  nueva_alfa, nueva_beta = fsolve_encontrar_x1_alfa_y_x1_beta(t_actual)
  x_alfa.append(nueva_alfa)
  x_beta.append(nueva_beta)

plt.plot(x_alfa, t)
plt.plot(x_beta, t)
plt.plot(global_x_experimental, global_t_experimental, "s")
plt.show()


x_calc = []

i = indice_del_maximo(global_t_experimental)

for t_actual in global_t_experimental[:i]:
  nueva_alfa, nueva_beta = fsolve_encontrar_x1_alfa_y_x1_beta(t_actual)
  x_calc.append(nueva_alfa)

for t_actual in global_t_experimental[i:]:
  nueva_alfa, nueva_beta = fsolve_encontrar_x1_alfa_y_x1_beta(t_actual)
  x_calc.append(nueva_beta)

print(rmsep(x_calc, global_x_experimental))

# De aqui para abajo es codigo en cuarantena
"""
def newton_rapson_f1_derivada_1_alfa(x1alfa, x1beta, t):
  tau, G = mezcla_calcular_funciones_de_t(t)

  tau12, tau21 = tau
  g12, g21 = G

  p1 = (1 - x1alfa) * (1 - x1alfa)
  p2 = -2 * tau21 * (1 - g21) * g21 * g21
  p3 = np.power((g21 * (1 - x1alfa) + x1alfa), 3)
  p4 = -2 * (g12 - 1) * g12 * tau12
  p5 = np.power((g12 * x1alfa - x1alfa + 1), 3)

  p6  = -2 * (1 - x1alfa)
  p7 = tau21 * g21 * g21
  p8 = np.power(g21 * (1 - x1alfa) + x1alfa, 2)
  p9 = g12 * tau12
  p10 = np.power(g12 * x1alfa - x1alfa + 1, 2)
  
  return p1 * (p2 / p3 + p4 / p5) + p6 * (p7 / p8 + p9 / p10) + 1 / x1alfa


def newton_rapson_f1_derivada_1_beta(x1alfa, x1beta, t):
  tau, G = mezcla_calcular_funciones_de_t(t)

  tau12, tau21 = tau
  g12, g21 = G

  p1 = (1 - x1beta) * (1 - x1beta)
  p2 = -2 * tau21 * (1 - g21) * g21 * g21
  p3 = np.power((g21 * (1 - x1beta) + x1beta), 3)
  p4 = -2 * (g12 - 1) * g12 * tau12
  p5 = np.power((g12 * x1beta - x1beta + 1), 3)

  p6  = -2 * (1 - x1beta)
  p7 = tau21 * g21 * g21
  p8 = np.power(g21 * (1 - x1beta) + x1beta, 2)
  p9 = g12 * tau12
  p10 = np.power(g12 * x1beta - x1beta + 1, 2)
  
  return p1 * (p2 / p3 + p4 / p5) + p6 * (p7 / p8 + p9 / p10) - 1 / x1beta


def newton_rapson_f2_derivada_1_alfa(x1alfa, x1beta, t):
  tau, G = mezcla_calcular_funciones_de_t(t)

  tau12, tau21 = tau
  g12, g21 = G

  p1 = x1alfa * x1alfa
  p2 = -2 * tau12 * (g12 - 1) * g12 * g12
  p3 = np.power((g12 * x1alfa - x1alfa + 1), 3)
  p4 = -2 * (1 - g21) * g21 * tau21
  p5 = np.power((g21 * (1 - x1alfa) + x1alfa), 3)

  p6  = 2 * x1alfa
  p7 = tau12 * g12 * g12
  p8 = np.power((g12 * x1alfa - x1alfa + 1), 2)
  p9 = g12 * tau12
  p10 = np.power((g21 * (1 - x1alfa) + x1alfa), 2)
  
  return p1 * (p2 / p3 + p4 / p5) + p6 * (p7 / p8 + p9 / p10) + 1 / (1 - x1alfa)


def newton_rapson_f2_derivada_1_beta(x1alfa, x1beta, t):
  tau, G = mezcla_calcular_funciones_de_t(t)

  tau12, tau21 = tau
  g12, g21 = G

  p1 = x1beta * x1beta
  p2 = -2 * tau12 * (g12 - 1) * g12 * g12
  p3 = np.power((g12 * x1beta - x1beta + 1), 3)
  p4 = -2 * (1 - g21) * g21 * tau21
  p5 = np.power((g21 * (1 - x1beta) + x1beta), 3)

  p6  = 2 * x1beta
  p7 = tau12 * g12 * g12
  p8 = np.power((g12 * x1beta - x1beta + 1), 2)
  p9 = g12 * tau12
  p10 = np.power((g21 * (1 - x1beta) + x1beta), 2)
  
  return p1 * (p2 / p3 + p4 / p5) + p6 * (p7 / p8 + p9 / p10) - 1 / (1 - x1beta)


def newton_rapson_determinante(x1alfa, x1beta, t):
  df1dx = newton_rapson_f1_derivada_1_alfa(x1alfa, x1beta, t)
  df1dy = newton_rapson_f1_derivada_1_beta(x1alfa, x1beta, t)
  df2dx = newton_rapson_f2_derivada_1_alfa(x1alfa, x1beta, t)
  df2dy = newton_rapson_f2_derivada_1_beta(x1alfa, x1beta, t)
  return np.linalg.det([[df1dx, df2dy], [df2dx, df2dy]])


def newton_rapson_delta_x1alfa(x1alfa, x1beta, t):
  J = newton_rapson_determinante(x1alfa, x1beta, t)
  df2dy = newton_rapson_f2_derivada_1_beta(x1alfa, x1beta, t)
  df1dy = newton_rapson_f1_derivada_1_beta(x1alfa, x1beta, t)
  a = -funcion_objetivo_1(x1alfa, x1beta, t) * df2dy
  b = funcion_objetivo_2(x1alfa, x1beta, t) * df1dy
  return (a + b) / J


def newton_rapson_delta_x1beta(x1alfa, x1beta, t):
  J = newton_rapson_determinante(x1alfa, x1beta, t)
  df2dx = newton_rapson_f2_derivada_1_alfa(x1alfa, x1beta, t)
  df1dx = newton_rapson_f1_derivada_1_alfa(x1alfa, x1beta, t)
  a = -funcion_objetivo_2(x1alfa, x1beta, t) * df1dx
  b = funcion_objetivo_1(x1alfa, x1beta, t) * df2dx
  return (a + b) / J


def newton_rapson_encontrar_x1_alfa_x1_beta(t):
  x1alfa_pasado = 0
  x1beta_pasado = 0
  x1alfa_actual = 0.2221
  x1beta_actual = 0.4978

  def seguir(err=5e-5):
    err_x1alfa = abs((x1alfa_actual - x1alfa_pasado) / x1alfa_actual) * 100
    err_x1beta = abs((x1beta_actual - x1beta_pasado) / x1beta_actual) * 100
    return err_x1alfa > err and err_x1beta > err

  while seguir():
    x1alfa_pasado = x1alfa_actual
    x1beta_pasado = x1beta_actual
    x1alfa_actual = x1alfa_pasado + newton_rapson_delta_x1alfa(x1alfa_pasado, x1beta_pasado, t)
    x1beta_actual = x1beta_pasado + newton_rapson_delta_x1beta(x1alfa_pasado, x1beta_pasado, t)
    print(x1alfa_pasado, x1beta_pasado)
    print(x1alfa_actual, x1beta_actual)

  return x1alfa_actual, x1beta_actual


def f_obj(x):
  alfa = newton_rapson_f1(x[0], x[1], 347.89)
  beta = newton_rapson_f2(x[0], x[1], 347.89)
  return alfa, beta

r = fsolve(f_obj, [0.2221, 0.4978])

print(r)
print(f_obj(r))
"""


