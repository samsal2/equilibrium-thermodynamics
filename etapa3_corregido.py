# Codigo corregido de la etapa 3

###############################################################################
# Librerias utilizadas
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import root
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
global_r_j_molk = 8.314


global_t_experimental = [334.24, 
                         345.48,
                         349.16, 
                         350.52,
                         350.44,
                         349.70,
                         346.09,
                         336.01,
                         327.71]
                         #299.71]

global_x_experimental = [0.0991, 
                         0.1669,
                         0.2374, 
                         0.3013,
                         0.4026,
                         0.4984,
                         0.5964,
                         0.7245,
                         0.7640]
                         #0.8919]
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


def nrtl_lny_sistema_binario_substancia_1(x, tau, g):
  x1, x2 = x
  t12, t21 = tau
  g12, g21 = g 


  # Precalculando valores
  a = (t21 * g21 * g21) / ((x1 + x2 * g21) * (x1 + x2 * g21))
  b = (t12 * g12) / ((x2 + x1 * g12) * (x2 + x1 * g12))

  return x2 * x2 * (a + b)


def nrtl_lny_sistema_binario_substancia_2(x, tau, g):
  x1, x2 = x
  t12, t21 = tau
  g12, g21 = g 

  # Precalculando valores
  a = (t12 * g12 * g12) / ((x2 + x1 * g12) * (x2 + x1 * g12))
  b = (t21 * g21) / ((x1 + x2 * g21) * (x1 + x2 * g21))

  return x1 * x1 * (a + b)



def nrtl_tau(delta_g, r, t):
  return delta_g / (r * t)


def nrtl_G(alfa_ji, tau_ji):
  return np.exp(-alfa_ji * tau_ji)


def nrtl_delta_g(aij, bij, cij, t, tc):
  t_diff = tc - t
  return aij + bij * t_diff + cij * t_diff * t_diff


def mezcla_calcular_nrtl_delta_g_12(t):
  a = global_parametro_interaccion_A12
  b = global_parametro_interaccion_B12
  c = global_parametro_interaccion_C12
  tc = global_tc_para_mezcla
  return nrtl_delta_g(a, b, c, t, tc)


def mezcla_calcular_nrtl_delta_g_21(t):
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

  return alfa - beta - np.log(x1beta / x1alfa)
  

def funcion_objetivo_2(x1alfa, x1beta, t):
  tau, G = mezcla_calcular_funciones_de_t(t)

  alfa = nrtl_lny_sistema_binario_substancia_2([x1alfa, 1 - x1alfa], tau, G)
  beta = nrtl_lny_sistema_binario_substancia_2([x1beta, 1 - x1beta], tau, G)
  
  return alfa - beta - np.log((1 - x1beta) / (1 - x1alfa))

def encontrar_x1_alfa_y_x1_beta(t):
  def funcion_objetivo(x):
    x1, x2 = x
    obj1 = funcion_objetivo_1(x1, x2, t)
    obj2 = funcion_objetivo_2(x1, x2, t)
    return obj1, obj2

  result = root(funcion_objetivo, [0.0991, 0.7245], method="lm")
  
  if not result.success:
    raise RuntimeError(f"Culprit{t}")

  return result.x


def nrtl_G12_derivada_inv_t(t):
  tau, _ = mezcla_calcular_funciones_de_t(t)
  tau12, _ = tau

  aij = global_parametro_interaccion_A12
  bij = global_parametro_interaccion_B12
  cij = global_parametro_interaccion_C12

  a = -global_alfa_12 / global_r_j_molk
  tc = global_tc_para_mezcla
  b = aij + bij * tc + cij * tc * tc - cij * t * t

  return a * b * nrtl_G(global_alfa_12, tau12)


def nrtl_G21_derivada_inv_t(t):
  tau, _ = mezcla_calcular_funciones_de_t(t)
  _, tau21 = tau

  aij = global_parametro_interaccion_A21
  bij = global_parametro_interaccion_B21
  cij = global_parametro_interaccion_C21

  a = -global_alfa_12 / global_r_j_molk
  tc = global_tc_para_mezcla
  b = aij + bij * tc + cij * tc * tc - cij * t * t

  return a * b * nrtl_G(global_alfa_12, tau21)


@np.vectorize
def energia_en_exceso(x1, t):
  x2 = 1 - x1
  x = [x1, x2]

  tau, G = mezcla_calcular_funciones_de_t(t)
  tau12, tau21 = tau
  G12, G21 = G
  Gp12 = nrtl_G12_derivada_inv_t(t)
  Gp21 = nrtl_G21_derivada_inv_t(t)

  a = x1 * x2 * global_r_j_molk
  b = (x1 * tau21 * Gp21) / ((x1 + x2 * G21) * (x1 + x2 * G21))
  c = (x2 * tau12 * Gp12) / ((x2 + x1 * G12) * (x2 + x1 * G12))
  
  return a * (b + c)


###############################################################################
# Inicio del codigo 
###############################################################################


# Calculo de T para graficar
t_graf = np.linspace(global_tc_para_mezcla, 320, 128)


# Calculo de x_graf_alfa, x_graf_beta
x_graf_alfa = []
x_graf_beta = []

for t_actual in t_graf:
  nueva_alfa, nueva_beta = encontrar_x1_alfa_y_x1_beta(t_actual)
  x_graf_alfa.append(nueva_alfa)
  x_graf_beta.append(nueva_beta)

x_graf = list(reversed(x_graf_alfa))
x_graf.extend(x_graf_beta)

tmp_t_graf = t_graf
t_graf = list(reversed(t_graf))
t_graf.extend(tmp_t_graf)


# Graficando datos calculados contra experimentales

plt.figure(1)
plt.plot(x_graf, t_graf, "k", label="calculado")

"""
plt.plot(x_graf_alfa, tmp_t_graf, "k", label="calculado")
plt.plot(x_graf_beta, tmp_t_graf, "k", label="calculado")
"""

# Graficando datos experimentales
plt.plot(global_x_experimental, 
         global_t_experimental, 
         "ks", 
         label="experimental", 
         markerfacecolor='none')

# Encontrando el punto critico

t_c_graf = tmp_t_graf[0]
x_c_graf = np.average([x_graf_alfa[0], x_graf_beta[0]])

plt.plot(x_c_graf, 
         t_c_graf, 
         "kv", 
         label="punto crítico", 
         markerfacecolor='none')

print(x_c_graf, t_c_graf)

plt.legend()
plt.xlabel("$x_1$")
plt.ylabel("T (˚K)")
plt.xlim(0, 1)

# Graficando entalpia en exceso
plt.figure(2)
he = energia_en_exceso(x_graf, t_graf)
he_exp = energia_en_exceso(global_x_experimental, global_t_experimental)


plt.plot(x_graf, he, "k", label="entalpía en exceso con NRTL")
plt.xlabel("$x_1$")
plt.ylabel("$H^E$ (J/mol)")
plt.xlim(0, 1)
plt.legend()

# Calculando datos para comparar con los experimentales
x_calc_alfa = []
x_calc_beta = []

for t_actual in global_t_experimental:
  nueva_alfa, nueva_beta = encontrar_x1_alfa_y_x1_beta(t_actual)
  x_calc_alfa.append(nueva_alfa)
  x_calc_beta.append(nueva_beta)


x_calc = []
x_calc.extend(x_calc_alfa[:4])
x_calc.extend(x_calc_beta[4:])

he_calc = energia_en_exceso(x_calc, global_t_experimental)

print(f"x1 RMSEP: {rmsep(x_calc, global_x_experimental)}")

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
plt.show()



