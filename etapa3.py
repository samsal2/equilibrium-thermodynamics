##############################################################################
# Librerias utilizadas
###############################################################################


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from openpyxl import load_workbook

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
    x1, x2 = x
    obj1 = funcion_objetivo_1(x1, x2, t)
    obj2 = funcion_objetivo_2(x1, x2, t)
    return obj1, obj2

  return fsolve(funcion_objetivo, [0.0503, 0.8224]) 


def fraccion_molar_local_x12(x, t):
  tau, _ = mezcla_calcular_funciones_de_t(t)
  tau12, tau21 = tau
  x1, x2 = x

  num = x1 * np.exp(-global_alfa_12 * tau12)
  den = x2 + x1 * np.exp(-global_alfa_12 * tau12)

  return den / num


def fraccion_molar_local_x21(x, t):
  tau, _ = mezcla_calcular_funciones_de_t(t)
  tau12, tau21 = tau
  x1, x2 = x

  num = x2 * np.exp(-global_alfa_12 * tau21)
  den = x1 + x2 * np.exp(-global_alfa_12 * tau21)

  return den / num


def energia_de_gibbs_en_exceso(x, t):
  tau, _ = mezcla_calcular_funciones_de_t(t)
  tau12, tau21 = tau


  tauij = [[0, tau12], [tau12, 0]]
  xij = [[0, fraccion_molar_local_x12(x, t)], 
         [fraccion_molar_local_x21(x , t), 0]]
  
  s = 0

  for i, xi in enumerate(x):
    for j, xj in enumerate(x):
      s += xi * xij[i][j] * tauij[i][j]

  return global_r_j_molk * t * s
  
  
###############################################################################
# 
###############################################################################

# Calculo de T para graficar
t_graf = np.linspace(global_tc_para_mezcla * 1.001, 300, 1024)

# Calculo de x_graf_alfa, x_graf_beta
x_graf_alfa = []
x_graf_beta = []

for t_actual in t_graf:
  nueva_alfa, nueva_beta = fsolve_encontrar_x1_alfa_y_x1_beta(t_actual)
  x_graf_alfa.append(nueva_alfa)
  x_graf_beta.append(nueva_beta)

# Graficando datos calculados contra experimentales
plt.figure(1)
plt.plot(x_graf_alfa, t_graf, label="alfa")
plt.plot(x_graf_beta, t_graf, label="beta")

# Graficando datos experimentales
plt.plot(global_x_experimental, global_t_experimental, "s", label="experimental")


# Calculando datos para comparar con los experimentales
x_calc_alfa = []
x_calc_beta = []

for t_actual in global_t_experimental:
  nueva_alfa, nueva_beta = fsolve_encontrar_x1_alfa_y_x1_beta(t_actual)
  x_calc_alfa.append(nueva_alfa)
  x_calc_beta.append(nueva_beta)

# Imprimiendo resultados comparando datos experimentales con calculados
# a la consola

print("+--------+------+------+------+")
print("| T (˚K) |x exp | alfa | beta |")
print("+--------+------+------+------+")
for t_exp, x_exp, x_alfa, x_beta in zip(global_t_experimental, 
                                        global_x_experimental, 
                                        x_calc_alfa, 
                                        x_calc_beta):
  print("|{:.4f}|{:.4f}|{:.4f}|{:.4f}|".format(t_exp, x_exp, x_alfa, x_beta))
  print("+--------+------+------+------+")

# Guardando los datos en un archivo de excel para facil manipulación para
# presentar

wb = load_workbook(filename="etapa3_tablas.xlsx")
hoja = wb.active

hoja["A1"] = "T (˚K)"
hoja["B1"] = "x experimental"
hoja["C1"] = "x alfa calculada"
hoja["D1"] = "x beta calculada"

i = 2
for t_exp, x_exp, x_alfa, x_beta in zip(global_t_experimental, 
                                        global_x_experimental, 
                                        x_calc_alfa, 
                                        x_calc_beta):

  hoja[f"A{i}"] = t_exp
  hoja[f"B{i}"] = x_exp
  hoja[f"C{i}"] = x_alfa
  hoja[f"D{i}"] = x_beta
  i += 1


wb.save(filename="etapa3_tablas.xlsx")

# Mostrar graficas
plt.legend()
plt.grid()
plt.xlabel("x1")
plt.ylabel("T (˚K)")
plt.show()



