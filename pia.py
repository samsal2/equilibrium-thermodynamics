##############################################################################
#  ________  ___  ________                                       
# |\   __  \|\  \|\   __  \                                      
# \ \  \|\  \ \  \ \  \|\  \                                     
#  \ \   ____\ \  \ \   __  \                                    
#   \ \  \___|\ \  \ \  \ \  \                                   
#    \ \__\    \ \__\ \__\ \__\                                  
#     \|__|     \|__|\|__|\|__|                                  
#                                                                
#                                                                
#                                                                
#  _________  _______   ________  _____ ______   ________        
# |\___   ___\\  ___ \ |\   __  \|\   _ \  _   \|\   __  \       
# \|___ \  \_\ \   __/|\ \  \|\  \ \  \\\__\ \  \ \  \|\  \      
#      \ \  \ \ \  \_|/_\ \   _  _\ \  \\|__| \  \ \  \\\  \     
#       \ \  \ \ \  \_|\ \ \  \\  \\ \  \    \ \  \ \  \\\  \    
#        \ \__\ \ \_______\ \__\\ _\\ \__\    \ \__\ \_______\   
#         \|__|  \|_______|\|__|\|__|\|__|     \|__|\|_______|   
#                                                                
#                                                                
#                                                                
#  ________  _______   ___               _______   ________      
# |\   ___ \|\  ___ \ |\  \             |\  ___ \ |\   __  \     
# \ \  \_|\ \ \   __/|\ \  \            \ \   __/|\ \  \|\  \    
#  \ \  \ \\ \ \  \_|/_\ \  \            \ \  \_|/_\ \  \\\  \   
#   \ \  \_\\ \ \  \_|\ \ \  \____        \ \  \_|\ \ \  \\\  \  
#    \ \_______\ \_______\ \_______\       \ \_______\ \_____  \ 
#     \|_______|\|_______|\|_______|        \|_______|\|___| \__\
#                                                           \|__|
###############################################################################
# Importar librerias y metodos
###############################################################################


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, root_scalar
from numpy.polynomial.polynomial import Polynomial


###############################################################################
# Declaraci??n de constantes globales
###############################################################################


global_x1_experimentales = [0.0000, 0.0037, 0.0102, 0.0161, 0.0207,
                            0.0314, 0.0431, 0.0613, 0.0854, 0.1263,
                            0.1811, 0.2334, 0.3217, 0.3805, 0.4201,
                            0.4746, 0.5420, 0.5716, 0.6164, 0.6509,
                            0.6793, 0.7259, 0.7575, 0.8171, 0.8744,
                            0.9033, 0.9264, 0.9497, 0.9594, 0.9707,
                            0.9804, 0.9895, 1]

global_y1_experimentales = [0.0000, 0.0882, 0.1567, 0.2363, 0.2794, 0.3391, 
                            0.3794, 0.4306, 0.4642, 0.4921, 0.5171, 0.5288, 
                            0.5450, 0.5538, 0.5590, 0.5673, 0.5783, 0.5821, 
                            0.5908, 0.5990, 0.6067, 0.6216, 0.6346, 0.6681, 
                            0.7181, 0.7525, 0.7896, 0.8368, 0.8599, 0.8916, 
                            0.9222, 0.9558, 1]

global_p_experimentales = [29.894, 32.744, 35.358, 38.587, 40.962,
                           44.231, 46.832, 50.488, 53.224, 55.571,
                           57.454, 58.427, 59.402, 59.802, 60.015,
                           60.242, 60.416, 60.443, 60.416, 60.350,
                           60.215, 59.868, 59.482, 58.321, 56.213,
                           54.692, 53.037, 51.009, 50.048, 48.767,
                           47.540, 46.232, 44.608]

global_t_kelvin = 45 + 273.15
global_tc_kelvin = [239.45 + 273.15, 289.05 + 273.15]
global_pc_kpa = [7953.79201, 4924.2357]
global_cuenta_de_figuras = 1
global_numero_figura_general = 10
global_vl_cm3_mol = [40.5, 89.4]
global_r_j_molk = 8.314

# Las referencias son las substancias liquidas
global_t_referencia_kelvin = 45 + 273.15


global_cp_metanol_l_kj_molc = Polynomial([75.86e-3, 16.85e-5])

global_cp_metanol_v_kj_molc = Polynomial([42.93e-3, 8.301e-5, 
                                          -1.87e-8, -8.03e-12])

global_cp_benceno_l_kj_molc = Polynomial([126.5e-3, 23.4e-5])

global_cp_benceno_v_kj_molc = Polynomial([74.06e-3, 32.95e-5, 
                                          -25.2e-8, 77.57e-12])

global_hv_metanol_j_mol = 38.58 * 1000
global_tb_metanol_kelvin = 78.5 + 273.15
global_hv_benceno_j_mol = 30.765 * 1000
global_tb_benceno_kelvin = 80.10 + 273.15


###############################################################################
# Declaraci??n de metodos
###############################################################################


def de_kpa_a_psia(p):
  """
  Funcion para convertir la presi??n de kPa a psia

  Parametros
  ----------
  p: float
    Presi??n (kPa)


  Devuelve
  --------
  float
    Presi??n (psia)
  """

  return p * 0.145038


def rmsep(v_calculado, v_experimental):
  """
  Funcion para calcular el rmsep

  Parametros
  ----------
  v_calculado: list[float]
    Lista con los valores calculados
  v_experimental: list[float]
    Lista con los valores experimentales

  Devuelve
  --------
  float
    El rmsep calculado
  """

  # Si assert evalua a falso el programa se para
  # Asegurarse que las longitudes del experimental y el observado sean iguales
  assert len(v_calculado) == len(v_experimental)

  # Encontar las cantidad de elementos
  n = len(v_calculado)

  # Se inicia la suma
  s = 0

  for v_obs, v_exp in zip(v_calculado, v_experimental):
    # Se saca la diferencia
    diff = v_obs - v_exp

    # Se suma el cuadrado de la diferencia
    s += diff * diff

  return np.sqrt(s / n)


def antoine_smith_presion_saturada(a, b, c, t):
  """
  Parametros
  ----------
  a : float
    Coeficiente A de la ecuaci??n de antoine
  b : float
    Coeficiente B de la ecuaci??n de antoine
  c : float
    Coeficiente C de la ecuaci??n de antoine
  t : float
    Temperatura en ??C

  Devuelve
  --------
  float
    presi??n saturada en kpa
  """
  return np.exp(a - b / (t + c))


def antoine_smith_metanol_presion_saturada(t):
  """
  Parametros
  ----------
  t : float
    Temperatura en ??K

  Devuelve
  --------
  float
    Presi??n saturada del metanol en kPa
  """
  return antoine_smith_presion_saturada(16.5785, 3638.27, 239.500, t - 273.15)


def antoine_smith_benceno_presion_saturada(t):
  """
  Parametros
  ----------
  t : float
    Temperatura en ??K

  Devuelve
  --------
  float
    Presi??n saturada del benceno en kPa
  """
  return antoine_smith_presion_saturada(13.7819, 2726.81, 217.572, t - 273.15)


def redlich_kwong_equacion_cubica(a, b, p, z):
  """
  Funcion que representa la ecuac??on 4-41 del seader

  Z^3 - Z^2 + BP(A^2/B - BP - 1)Z - A^2/B(BP)^2 = f(Z)

  Parametros
  ----------
  a : float
    Regla de mezcla A, descrita por la ecuaci??n 4-42 del seader
  b : float
    Reglas de mezcla B, descrita por la ecua??on 4-43 del seader
  p : float
    Presi??n
  z : float (kPa)
    Factor de compresibilidad

  Devuelve
  --------
  float
    El resultado de la funci??n f(Z)
  """

  # Se recalcula la p como psia
  p_psia = de_kpa_a_psia(p)

  # Precalcular variables para representar la ecuaci??n como:
  # Z^3 - Z^2 + a1 * Z + a0 = f(Z)

  a0 = -a * a / b * (b * p_psia) * (b * p_psia)
  a1 = b * p_psia * (a * a / b - b * p_psia - 1)
  return z * z * z - z * z + a1 * z + a0


def redlich_kwong_equacion_cubica_derivada(a, b, p, z):
  """
  metodo que representa al derivada de la ecuaci??n 4-41 del seader

  f'(Z) = 3Z^2 - 2Z + BP(((A^2) / B) - BP - 1)

  Parametros
  ----------
  a : float
    Regla de mezcla A, descrita por la ecuaci??n 4-42 del seader
  b : float
    Reglas de mezcla B, descrita por la ecua??on 4-43 del seader
  p : float
    Presi??n (kPa)
  z : float
    Factor de compresibilidad

  Devuelve
  --------
  float
    El resultado de la funci??n f'(Z)
  """
  # Se recalcula la p como psia
  p_psia = de_kpa_a_psia(p)

  c = b * p_psia * ((a * a) / b - b * p_psia - 1)
  return 3 * z * z - 2 * z + c


def redlich_kwong_resolver_equacion_cubica(a, b, p, z0=1, err=5e-10):
  """
  Funcion para encontrar el valor de Z

  Parametros
  ----------
  a : float
    Regla de mezcla A, descrita por la ecuaci??n 4-42 del seader
  b : float
    Reglas de mezcla B, descrita por la ecua??on 4-43 del seader
  p : float
    Presi??n (kPa)
  z0 : float
    Z inicial
    Si se quiere encontrar el valor de Z liquido, Z0 tiene que ser cercano a 0
    Si se quiere encontrar el valor de Z vapor, Z0 tiene que ser cercano a 1
  err : float
    El error relativo

  Devuelve
  --------
  float
    El valor de Z encontrado
  """

  # Se copia el valor de z0 inicial a la z_pasada
  z_pasada = z0

  # Funcion de newton rapson
  # z_i+1 = z_i - f(z_i) / f'(z_i)

  # Primera iteraci??n de newton rapson para comenzar el ciclo
  z_actual = z_pasada - redlich_kwong_equacion_cubica(a, b, p, z_pasada) /\
      redlich_kwong_equacion_cubica_derivada(a, b, p, z_pasada)

  while abs((z_actual - z_pasada) / z_actual * 100) > err:
    # Se guarda el valor de z en z_pasada
    z_pasada = z_actual

    # Se calcula la siguiente iteraci??n de newton rapson
    z_actual = z_pasada - redlich_kwong_equacion_cubica(a, b, p, z_pasada) /\
        redlich_kwong_equacion_cubica_derivada(a, b, p, z_pasada)

  return z_actual


def regla_de_mezcla_a(a, y):
  """
  Regla de mezclado del parametro A descrita por la ecuaci??n 4-42 del seader

  A = ??Aixi 

  Parametros
  ----------
  A : list[float]
    Lista con los valores de A en cada posici??n de i
  y : list[float]
    Lista con los valores de la composici??n de la mezcla

  Devuelve
  --------
  float
    Parametro A de la ecuaci??n Redlich-Kwong
  """

  # Si assert evalua a falso el programa se para
  # Asegurarse que la lista de A y x tengan la misma longitud
  assert len(a) == len(y)

  # valor inicial de la sumatoria
  s = 0

  # sumar los productos
  for ai, yi in zip(a, y):
    s += ai * yi

  return s


def regla_de_mezcla_b(b, y):
  """
  Regla de mezclado del parametro B descrita por la ecuaci??n 4-43 del seader

  B = ??Biyi 

  Parametros
  ----------
  b : list[float]
    Lista con los valores de B en cada posici??n de i
  x: list[float]
    Lista con los valores de la composici??n de la mezcla

  Devuelve
  --------
  float
    Parametro B para la ec de redlich kwong del factor de compresibilidad
  """

  # Si assert evalua a falso el programa se para
  # Asegurarse que la lista de B y x tengan la misma longitud
  assert len(b) == len(y)

  # valor inicial de la sumatoria
  s = 0

  # sumar los productos
  for bi, yi in zip(b, y):
    s += bi * yi

  return s


def redlich_kwong_ai(pc, tr):
  """
  Valor de A para la substancia i segun la ecuaci??n 4-44 del seader

  Parametros
  ----------
  pc : float
    Presi??n critica (kPa)

  tr : float
    Temperatura reducida  (??K)

  Devuelve
  --------
  float
    Valor de Ai
  """
 
  # El seader toma la Pc como psias 
  pc_psia = de_kpa_a_psia(pc)

  return np.sqrt(0.4278 / (pc_psia * np.power(tr, 2.5)))


def redlich_kwong_bi(pc, tr):
  """
  Valor de A para la substancia i segun la ecuaci??n 4-44 del seader

  Parametros
  ----------
  pc : float
    Presi??n critica (kPa)

  tr : float
    Temperatura reducida  (??K)

  Devuelve
  --------
  float
    Valor de Ai
  """

  # El seader toma la Pc como psias 
  pc_psia = de_kpa_a_psia(pc)

  return 0.0867 / (pc_psia * tr)


def coeficiente_de_fugacidad_para_ecuacion_cubica(z, a, ai, b, bi, p):
  """
  Funcion para calcular la fugacidad descrita de la ecuaci??n 4-72 del seader

  ??i = e^[(Z - 1)Bi/B - ln(Z - BP) - A^2/B(2Ai/A - Bi/B)ln(1 + BP/Z)]

  Parametros
  ----------
  z : float
    Factor de compresiblidad
  a : float
    Regla de mezcla A, descrita por la ecuaci??n 4-42 del seader
  ai : float
    Valor de A para la substancia i
  b : float
    Reglas de mezcla B, descrita por la ecua??on 4-43 del seader
  bi : float
    Valor de B para la substancia i
  p : float
    Presi??n (kPa)

  Devuelve
  --------
  float
    Fugacidad de la substancia i liquida
  """

  # El seader toma la p como psias 
  p_psia = de_kpa_a_psia(p)

  # La funcion se va a separar para simplificarla de la siguiente manera:
  # ??iL = e^[(Z - 1)Bi/B - ln(Z - BP) - A^2/B(2Ai/A - Bi/B)ln(1 + BP/Z)]
  # d = (Z - 1)Bi/B
  # e = -ln(Z - BP)
  # f = -A^2/B
  # g = 2Ai/A - Bi/B
  # h = ln(1 + BP/Z)
  # ??iL = e^(d + e + f * g * h)

  # Precalculando valores para simplificar
  d = (z - 1) * bi / b
  e = -np.log(z - b * p_psia)
  f = -a * a / b
  g = 2 * ai / a - bi / b
  h = np.log(1 + b * p_psia / z)
  
  return np.exp(d + e + f * g * h)


def redlich_kwong_calcular_compresibilidad(y, p, t):
  """
  Funcion para calcular la compresibilidad de una substancia i en una mezcla
  utilizando los datos globales

  Parametros
  ----------
  y  : list[float]
    Una lista con la composici??n de la mezcla
  p : float
    La presi??n (kPa)
  t : float
    La temperatura (??K)

  Devuelve
  --------
  float
    La fugacidad de la substancia i
  """

  tc = global_tc_kelvin
  pc = global_pc_kpa

  # Si assert evalua a falso el programa se para
  # Asegurarse de que las longitudes de las listas sean las mismas
  assert len(tc) == len(pc) == len(y)

  # Calcular lista de tr
  tr = [t / tci for tci in tc]

  # calcular lista de Ai y Bi
  lista_ai = []
  lista_bi = []
  for pci, tri in zip(pc, tr):
    lista_ai.append(redlich_kwong_ai(pci, tri))
    lista_bi.append(redlich_kwong_bi(pci, tri))

  # Calcular A y B por reglas de mezclado
  a = regla_de_mezcla_a(lista_ai, y)
  b = regla_de_mezcla_b(lista_bi, y)

  # Calcular la z
  return redlich_kwong_resolver_equacion_cubica(a, b, p)


def redlich_kwong_coeficiente_de_fugacidad_substancia_i(pc, tc, y, p, t, i):
  """
  Funcion para calcular la fugacidad de una substancia i en una mezcla

  Parametros
  ----------
  pc : list[float]
    Una lista con las presi??nes criticas para i (kPa)
  tc : list[float]
    Una lista con las temperatura criticas para i (??K)
  y  : list[float]
    Una list con la composici??n de la mezcla
  p : float
    La presi??n (kPa)
  t : float
    La temperatura (??K)
  i : int
    Indice de la substancia comenzando en 1

  Devuelve
  --------
  float
    La fugacidad de la substancia i
  """

  # Si assert evalua a falso el programa se para
  # Asegurarse de que las longitudes de las listas sean las mismas
  assert len(tc) == len(pc) == len(y)

  # Calcular lista de tr
  tr = [t / tci for tci in tc]

  # calcular lista de Ai y Bi
  lista_ai = []
  lista_bi = []
  for pci, tri in zip(pc, tr):
    lista_ai.append(redlich_kwong_ai(pci, tri))
    lista_bi.append(redlich_kwong_bi(pci, tri))

  # Calcular A y B por reglas de mezclado
  a = regla_de_mezcla_a(lista_ai, y)
  b = regla_de_mezcla_b(lista_bi, y)

  # Calcular la z
  z = redlich_kwong_resolver_equacion_cubica(a, b, p)

  # Inicializar el valor de a y b de la substancia i
  # las listas en python comienzar a contar desde 0
  ai = lista_ai[i - 1]
  bi = lista_bi[i - 1]

  return coeficiente_de_fugacidad_para_ecuacion_cubica(z, a, ai, b, bi, p)


def redlich_kwong_fugacidad_metanol(y, p, t):
  """
  Funcion para calcular el coefficiente de fugacidad del metanol
  en el sistema de metanol benceno

  Parametros
  ----------
  y : list[float]
    Lista con los valores de y1 y y2
  p : float
    Presi??n (kPa)
  t : float
    Temperatura (??K)

  Devuelve
  --------
  float
    El coeficiente de fugacidad del metanol
  """

  # Se guardan los valores criticos en variables con nombres mas peque??os
  # para su uso inmediato
  pc = global_pc_kpa
  tc = global_tc_kelvin

  # Se calcula el coeficiente de fugacidad de 1 (el metanol)
  return redlich_kwong_coeficiente_de_fugacidad_substancia_i(pc, tc, y, 
                                                             p, t, 1)


def redlich_kwong_fugacidad_benceno(y, p, t):
  """
  Funcion para calcular el coefficiente de fugacidad del benceno
  en el sistema de metanol benceno

  Parametros
  ----------
  y : list[float]
    Lista con los valores de y1 y y2
  p : float
    Presi??n (kPa)
  t : float
    Temperatura (??K)

  Devuelve
  --------
  float
    El coeficiente de fugacidad del benceno
  """

  # Se guardan los valores criticos en variables con nombres mas peque??os
  # para su uso inmediato
  pc = global_pc_kpa
  tc = global_tc_kelvin

  # Se calcula el coeficiente de fugacidad de 2 (el benceno)
  return redlich_kwong_coeficiente_de_fugacidad_substancia_i(pc, tc, y, 
                                                             p, t, 2)


def redlich_kwong_phi_mayus_metanol(y, p, t):
  """
  Funcion para calcular phi mayus del metanol para el modelo evl gamma-phi

  Parametros
  ----------
  y : list[float]
    Lista con los valores de y1 y y2
  p : float
    Presi??n (kPa)
  t : float
    Temperatura (??K)

  Devuelve
  --------
  float
    El coeficiente de fugacidad del benceno
  """
  
  # Se calcula el coeficiente de fugacidad
  phi = redlich_kwong_fugacidad_metanol(y, p, t)

  # Se calcula la presi??n saturada del metanol
  p_sat = antoine_smith_metanol_presion_saturada(t)

  # Se calcula la phi saturada
  phi_sat = redlich_kwong_fugacidad_metanol(y, p_sat, t)

  # NOTA: se omitio el factor de poynting
  return phi / phi_sat


def redlich_kwong_phi_mayus_benceno(y, p, t):
  """
  Funcion para calcular phi mayus del benceno para el modelo evl gamma-phi

  Parametros
  ----------
  y : list[float]
    Lista con los valores de y1 y y2
  p : float
    Presi??n (kPa)
  t : float
    Temperatura (??K)

  Devuelve
  --------
  float
    El coeficiente de fugacidad del benceno
  """

  # Se calcula el coeficiente de fugacidad
  phi = redlich_kwong_fugacidad_benceno(y, p, t)

  # Se calcula la presi??n saturada del benceno
  p_sat = antoine_smith_benceno_presion_saturada(t)

  # Se calcula la phi saturada
  phi_sat = redlich_kwong_fugacidad_benceno(y, p_sat, t)

  # NOTA: se omitio el factor de poynting
  return phi / phi_sat


def wilson_gamma_1(x, a12, a21):
  """
  Ecuaci??n 5-28 del seader para calcular la gamma 1 de wilson

  gamma = e^[-ln(x1 + A12x2) 
              + x2[A12 / (x1 + A12x2) - A21 / (x2 + A12x1)]]

  Parametros
  ----------
  x : list[float]
    Lista con los valores de x1 y x2
  a12 : float
    Valor de A12 de wilson
  a21 : float
    Valore de A21 de wilson

  Devuelve
  --------
  float
    gamma1
  """

  # Se inicializa el valor de x1 y x2
  x1, x2 = x

  # Se separa la ecuaci??n de la siguiente manera
  # gamma = e^[-ln(x1 + A12x2) 
  #            + x2[A12 / (x1 + A12x2) - A21 / (x2 + A12x1)]]
  # a = x1 + A12x2
  # b = A12 / (x1 + A12x2)
  # c = A21 / (x2 + A12x1)
  # gamma = e^[-ln(a) + x2 * (b - c)]]

  # Se precalculan los valores
  a = x1 + x2 * a12
  b = a12 / a
  c = a21 / (x2 + x1 * a21)

  # Se calcula la gamma
  return np.exp(-np.log(a) + x2 * (b - c))


def wilson_gamma_2(x, a12, a21):
  """
  Ecuaci??n 5-28 del seader para calcular la gamma 2 de wilson

  gamma = e^[-ln(x2 + A21x1) 
              - x1[A12 / (x1 + A12x2) - A21 / (x2 + A12x1)]]

  Parametros
  ----------
  x : list[float]
    Lista con los valores de x1 y x2
  a12 : float
    Valor de A12 de wilson
  a21 : float
    Valore de A21 de wilson

  Devuelve
  --------
  float
    gamma1
  """
  # Se inicializa el valor de x1 y x2
  x1, x2 = x

  # Se separa la ecuaci??n de la siguiente manera
  # gamma = e^[-ln(x2 + A12x1) 
  #            - x1[A12 / (x1 + A12x2) - A21 / (x2 + A12x1)]]
  # a = x2 + A12x1
  # b = A12 / (x1 + A12x2)
  # c = A21 / (x2 + A12x1)
  # gamma = e^[-ln(a) - x1 * (b - c)]]
  
  # Se precalculan los valores
  a = x2 + x1 * a21
  b = a12 / (x1 + x2 * a12)
  c = a21 / a
  return np.exp(-np.log(a) - x1 * (b - c))


def bublp_raoult(x, t):
  """
  Funcion para calcular bublp con rault para el sistema de 
  metanol y benceno

  Parametros
  ----------
  x : list[float]
    Lista con los valores de x1 y x2
  t : float
    Temperatura (??K)

  Devuelve
  -------
  list[Union[float, list]]
    El primer elemento de la lista es la presi??n, el segundo elemento es una
    lista con los valores de y1 y y2 
  """
  
  # Se inicializa x1 y x2 
  x1, x2 = x

  # Se calculan las presiones saturadas
  p_sat_1 = antoine_smith_metanol_presion_saturada(t)
  p_sat_2 = antoine_smith_benceno_presion_saturada(t)

  # Se calcula la presi??n
  p = x1 * p_sat_1 + x2 * p_sat_2

  # Se calculan las y
  y1 = x1 * p_sat_1 / p
  y2 = x2 * p_sat_2 / p

  return p, [y1, y2]


def bublp_raoult_mod(x, gamma, t):
  """
  Funcion para calcular bublp con raoult modificado para el sistema de 
  metanol y benceno

  Parametros
  ----------
  x : list[float]
    Lista con los valores de x1 y x2
  gamma : list[float]
    Lista con los valores de gamma precalculados para la temperatura correcta
  t : float
    Temperatura (??K)

  Devuelve
  -------
  list[Union[float, list]]
    El primer elemento de la lista es la presi??n, el segundo elemento es una
    lista con los valores de y1 y y2 
  """
  # NOTA: la razon por la que se piden las gammas precalculadas es para
  # permitir el uso de diferentes metodos para calcularas y la
  # facil comparacion de ambos metodos


  # Se inicializan las x
  x1, x2 = x

  # Se inicializan las gamma
  gamma1, gamma2 = gamma

  p_sat_1 = antoine_smith_metanol_presion_saturada(t)
  p_sat_2 = antoine_smith_benceno_presion_saturada(t)

  p = x1 * gamma1 * p_sat_1 + x2 * gamma2 * p_sat_2
  y1 = x1 * gamma1 * p_sat_1 / p
  y2 = x2 * gamma2 * p_sat_2 / p

  return p, [y1, y2]


def gamma_phi_p(x, gamma, p_sat, phi):
  """
  Funcion para calcular la presi??n segun la ecuaci??n 14.10 del smith

  Parametros
  ----------
  x : list[float]
    Lista con los valores de {xi...}
  gamma : list[float]
    Lista con los valores de {gamma_i...} precalculados para la 
    temperatura correcta
  p_sat : list[float]
    Valores de {Psat_i...} precalculados para la temperatura correcta
  phi : list[float]
    Valores de {??i...} precalculados para la presi??n y temperatura correcta

  Devuelve
  --------
  float
    Presi??n (unidades de p_sat)
  """
  
  # Asegurarse que las listas tengan el mismo tama??o
  assert len(x) == len(gamma) == len(p_sat) == len(phi)

  # Se inicializa la presi??n
  p = 0

  # Se aplica la sumatoria
  for x_i, gamma_i, p_sat_i, phi_i in zip(x, gamma, p_sat, phi):
    p += x_i * gamma_i * p_sat_i / phi_i

  return p


def gamma_phi_y(x, gamma, p_sat, phi, p):
  """
  Funcion para calcular los valores de y segun la ecuaci??n 14.8 del smith

  Parametros
  ----------
  x : list[float]
    Lista con los valores de {xi...}
  gamma : list[float]
    Lista con los valores de {gamma_i...} precalculados para la 
    temperatura correcta
  p_sat : list[float]
    Valores de {Psat_i...} precalculados para la temperatura correcta (kPa)
  phi : list[float]
    Valores de {??i...} precalculados para la presi??n y temperatura correcta
  p : float
    Presi??n (kPa)

  Devuelve
  --------
  list[float]
    Una lista con los valores de y
  """
  # Asegurarse que las listas tengan el mismo tama??o
  assert len(x) == len(gamma) == len(p_sat) == len(phi)

  # Se inicializa la lista de los valores de y
  y = []

  for x_i, gamma_i, p_sat_i, phi_i in zip(x, gamma, p_sat, phi):
    y.append(x_i * gamma_i * p_sat_i / (phi_i * p))

  return y


def bublp_gamma_phi(x, gamma, t, err=5e-2):
  """
  Funcion para calcular bublp con gamma-phi para el sistema de 
  metanol y benceno mostrado por el diagrama 14.1 del smith

  Parametros
  ----------
  x : list[float]
    Lista con los valores de x1 y x2
  gamma : list[float]
    Lista con los valores de gamma precalculados para la temperatura correcta
  t : float
    Temperatura (??K)
  err : float
    Error porcentual permitido en el calculo de bublp

  Devuelve
  -------
  list[Union[float, list]]
    El primer elemento de la lista es la presi??n, el segundo elemento es una
    lista con los valores de y1 y y2 
  """
  
  # Se inicializa phi
  phi = [1.0, 1.0]
  
  # Se calculan las presiones saturadas
  p_sat_1 = antoine_smith_metanol_presion_saturada(t)
  p_sat_2 = antoine_smith_benceno_presion_saturada(t)
  p_sat = [p_sat_1, p_sat_2]

  # En este punto segun el diagrama se calculan las gammas,
  # pero el metodo ya las captura. Esta en el que llama la funcion
  # en pasarlas a la temperatura correcta.
  # Esto se hace para facilitar la comparacion entre diferentes metodos
  # y para el calculo de minimos cuadrados para encontrar los coefficientes
  # de wilson

  p_pasada = 0
  
  # Se calcula la P segun la ecuac??on 14.10 del smith
  p = gamma_phi_p(x, gamma, p_sat, phi)

  # Se inicializa una lista de y afuera del scope de el loop
  y = []

  # Se compara la presi??n pasada y actual
  while abs((p - p_pasada) / p * 100) > err:
    # Se asigna la presi??n pasada para calcular la nueva
    p_pasada = p

    # Se calculan los valores de y segun la ecuaci??n 14.8 del smith
    y = gamma_phi_y(x, gamma, p_sat, phi, p)

    # Se calculan las phi de metanol y benceno
    phi[0] = redlich_kwong_phi_mayus_metanol(y, p, t)
    phi[1] = redlich_kwong_phi_mayus_benceno(y, p, t)

    # Se saca el nuevo valor de presi??n
    p = gamma_phi_p(x, gamma, p_sat, phi)

  return p, y


def calcular_wilson_a12_a21_gamma_phi(t):
  """
  Funcion para calcular los valores de A12 y A21 de wilson utilizando
  los valores globales experimentales y bublp de gamma phi


  Utiliza una funcion objetivo
  Fobj = ??(p_experimental - p_calculada) y la funcion minimizar para 
  encontrar A12 y A21, donde p_calculada se calcula usando bublp_gamma_phi

  Parametros
  ----------
  t : float
    Temperatura (??K)

  Devuelve
  --------
  list[float]
    Una lista con los valores de A12 y A21
  """
  
  # Se inicializa una lista con los valores de x1 experimentales omitiendo el
  # primer valor de x1 y el ultimo
  x1 = global_x1_experimentales[1:-1]

  # Se inicializa una lista con los valores de presi??n experimentales 
  # omitiendo el primer valor de presi??n y el ultimo
  p = global_p_experimentales[1:-1]

  # Se comienza la declaraci??n de la funcion objetivo
  def funcion_objetivo(a):
    # Se inicializan los valores de A12 y A21 que se este probando
    a12, a21 = a

    # Se inicializa la suma
    s = 0

    for x1_exp, p_exp in zip(x1, p):
      # Calcula la x2 experimental
      x2_exp = 1 - x1_exp
      
      # Se inicializa la lista de x1 y x2 experimentales
      x = [x1_exp, x2_exp]
      
      # Se calculan las gammas
      gamma1 = wilson_gamma_1(x, a12, a21)
      gamma2 = wilson_gamma_2(x, a12, a21)

      # Se inicializa la lista de gammas calculadas
      gamma = [gamma1, gamma2]

      # Se calcula la presi??n con bublp
      p_calculada, _ = bublp_gamma_phi(x, gamma, t)
      
      # Se calcula la diferencia
      diff = p_exp - p_calculada
      
      # Se saca el cuadrado de la diferencia y se le agrega a la sumatoria
      s += diff * diff

    return s

  # Restricciones para que los valores no se alejen mucho de 0
  # Sin estas el valor nunca converge
  def restringir_a12(a):
    return a[0]

  def restringir_a21(a):
    return a[1]
  
  restricciones = ({"type": "ineq", "fun": restringir_a12},
                   {"type": "ineq", "fun": restringir_a21})

  # Se minimiza la funcion objetivo y se devuelve el restulado
  resultado =  minimize(funcion_objetivo, [0.5, 1], constraints=restricciones)

  # Si se encontro un resultado se devuelve
  if resultado.success:
    return resultado.x

  # Si no se levanta una excepci??n
  raise RuntimeError("no se llego a un resultado")


def calcular_wilson_a12_a21_raoult_mod(t):
  """
  Funcion para calcular los valores de A12 y A21 de wilson utilizando
  los valores globales experimentales y bublp de raoult modificada


  Utiliza una funcion objetivo
  Fobj = ??(p_experimental - p_calculada) y la funcion minimizar para 
  encontrar A12 y A21, donde p_calculada se calcula usando bublp_raoult_mod

  Parametros
  ----------
  t : float
    Temperatura (??K)

  Devuelve
  --------
  list[float]
    Una lista con los valores de A12 y A21
  """
  # Se inicializa una lista con los valores de x1 experimentales omitiendo el
  # primer valor de x1 y el ultimo
  x1 = global_x1_experimentales[1:-1]

  # Se inicializa una lista con los valores de presi??n experimentales 
  # omitiendo el primer valor de presi??n y el ultimo
  p = global_p_experimentales[1:-1]

  # Se comienza la declaraci??n de la funcion objetivo
  def funcion_objetivo(a):
    # Se inicializan los valores de A12 y A21 que se este probando
    a12, a21 = a

    # Se inicializa la sumatoria
    s = 0

    for x1_exp, p_exp in zip(x1, p):
      # Calcula la x2 experimental
      x2_exp = 1 - x1_exp
      
      # Se inicializa la lista de x1 y x2 experimentales
      x = [x1_exp, x2_exp]

      # Se calculan las gammas
      gamma1 = wilson_gamma_1(x, a12, a21)
      gamma2 = wilson_gamma_2(x, a12, a21)

      # Se inicializa la lista de gammas calculadas
      gamma = [gamma1, gamma2]

      # Se calcula la presi??n con bublp
      p_calculada, _ = bublp_raoult_mod(x, gamma, t)

      # Se calcula la diferencia
      diff = p_exp - p_calculada

      # Se saca el cuadrado de la diferencia y se le agrega a la sumatoria
      s += diff * diff

    return s

  # Restricciones para que los valores no se alejen mucho de 0
  # Sin estas el valor nunca converge
  def restringir_a12(a):
    return a[0]

  def restringir_a21(a):
    return a[1]

  restricciones = ({"type": "ineq", "fun": restringir_a12},
                   {"type": "ineq", "fun": restringir_a21})

  # Se minimiza la funcion objetivo
  resultado = minimize(funcion_objetivo, [0.5, 1], constraints=restricciones)

  # Si se encontro un resultado se devuelve
  if resultado.success:
    return resultado.x

  # Si no se levanta una excepci??n
  raise RuntimeError("no se llego a un resultado")


def mostrar_datos(y_calc, p_calc, x_graf, y_graf, p_graf, titulo, hl=None, hv=None, marcador = ""):
  """
  Funcion para mostrar los calculados (faltan energias)

  Parametros
  ----------
  y_calc : list[float]
    Valores de y calculados en referencia a las x en global_x1_experimental
  p_calc : list[float]
    Valores de presi??n calculados en referencia con las presiones en
    global_p_experimental (kPa)
  x_graf : list[float]
    Lista con los valores de x para graficar
  y_graf : list[float]
    Lista con los valores de y para graficar
  p_graf : list[float]
    Lista con los valores de presi??n para graficar
  tiutlo : str
    Titulo a usar en la grafica
  """

  global global_cuenta_de_figuras

  # Se calcula el rmsep
  rmsep_calculado = rmsep(p_calc, global_p_experimentales)

  # Se imprime el resultado
  print(f"{titulo}: rmsep = {rmsep_calculado}")

  plt.figure(global_cuenta_de_figuras)
  # Se asigna el titulo a la grafica
  plt.title(titulo)

  # Se grafica el punto de burbuja
  plt.plot(x_graf, p_graf, label="punto de burbuja calculado")

  # Se grafica el punto de rocio
  plt.plot(y_graf, p_graf, label="punto de rocio calculado")

  # Se grafica el punto de burbuja experimental
  plt.plot(global_x1_experimentales, 
           global_p_experimentales, 
           "s", 
           label="punto de burbuja experimental")

  # Se grafica el punto de rocio experimental
  plt.plot(global_y1_experimentales, 
           global_p_experimentales, 
           "v", 
           label="punto de rocio experimental")

  # Al eje x se le nombra x1-y1
  plt.xlabel("$x_1,y_1$")
  
  # Al eje y se le nombra P (kPa)
  plt.ylabel("P (kPa)")

  # Se muestra la grafica
  plt.legend()

  # Se cambia la cuenta de figuras
  global_cuenta_de_figuras += 1

  
  # Si hay entalpias en exceso que graficar
  if hl is not None and hv is not None:
    # Se crea una nueva figura
    plt.figure(global_cuenta_de_figuras)

    # Se el asigna el titulo del metodo
    plt.title(titulo)

    # Se grafica
    plt.plot(x_graf, hl, "-", label="$H_{(l)}$")
    plt.plot(y_graf, hv, "-", label="$H_{(v)}$")

    # Se muestra la grafica
    plt.legend()
    plt.xlabel("$x_1, y_1$")
    plt.ylabel("$H (J / mol)$")

    # Se cambia la cuenta de figuras
    global_cuenta_de_figuras += 1

    plt.figure(global_numero_figura_general + 1)
    plt.plot(x_graf, hl, "-" + marcador, label=titulo + " $H_{(l)}$")
    plt.plot(y_graf, hv, "-" + marcador, label=titulo + " $H_{(v)}$")

    # Se muestra la grafica
    plt.legend()
    plt.xlabel("$x_1, y_1$")
    plt.ylabel("$H (J / mol)$")


  plt.figure(global_numero_figura_general)
  plt.title("Comparaci??n de todos los modelos")
  plt.plot(x_graf, p_graf, "-" + marcador, label=f"punto de burbuja {titulo}")
  plt.plot(y_graf, p_graf, "-" + marcador, label=f"punto de rocio {titulo}")

  # Al eje x se le nombra x1-y1
  plt.xlabel("$x_1,y_1$")
  
  # Al eje y se le nombra P (kPa)
  plt.ylabel("P (kPa)")

  plt.legend()

def mostrar_figuras():
  plt.show()


def flash_k(p, t):
  """
  Funcion para calcular los valores de K para flash
  
  Parametros
  ----------
  p: float
    Presi??n (kPa)
  t: float
    Temperatura (??K)

  Devuelve
  --------
  list[float]
    Lista con los valores de K1 y K2
  """

  p_sat_1 = antoine_smith_metanol_presion_saturada(t)
  p_sat_2 = antoine_smith_benceno_presion_saturada(t)
  return [p_sat_1 / p, p_sat_2 / p]


def normalizar_lista(l):
  """
  Funcion para normalizar una lista de numeros

  Parametros
  ----------
  l: list[float]
    Lista con numeros

  Devuelve
  --------
  list[float]
    List con los valores normalizados
  """

  # Se saca la suma
  s = sum(l)
    
  # Se saca la lista con valores normalizados
  return [v / s for v in l]


def bublp_flash(x, t):
  """
  Funcion para calcular el bublp con el flash

  Parametros
  ----------
  x: list[float]
    List con los valores de x1 y x2
  t: float
    Temperatura (??K)
 
  Devuelve
  --------
  list[Union[float, list]]
    Presi??n (kPa), [y1, y2]
  """

  # Se calculan las presiones saturadas
  p_sat_1 = antoine_smith_metanol_presion_saturada(t)
  p_sat_2 = antoine_smith_benceno_presion_saturada(t)
  p_sat = [p_sat_1, p_sat_2]

  # Se estiman y y P
  y_pasadas = [0.0, 0.0]
  y_actual = [0.5, 0.5]
  z = x
  p = sum([zi * p_sat_i for zi, p_sat_i in zip(z, p_sat)])

  # La funci??n objetivo descrita por el seader 7-18
  def funcion_objetivo(z, k):
    s = 0

    for zi, ki in zip(z, k):
      s += zi * ki

    return 1 - s

  # Lo mas cercano a un do while
  while True:
    # Se calculan las k
    k = flash_k(p, t)

    # Se estiman las y
    y_actual = [zi * ki for zi, ki in zip(z, k)]
    
    # Se normalizan
    y_actual = normalizar_lista(y_actual) 
    
    # Se checa la funcion objetivo
    if funcion_objetivo(z, k) < 1e-15 and funcion_objetivo(z, k) > -1e-15:
      break
  
    # Se vuelve a estimar la presi??n
    p = sum([zi * p_sat_i for zi, p_sat_i in zip(z, p_sat)])

  return p, y_actual


def wilson_lambdas(t, a12, a21):
  """
  Funcion para calcular las diferencias de lambdas para wilson
  para la mezcal metanol - benceno con las ecuaciones 5-39 y 5-40 del seader

  Parametros
  ----------
  t: float
    Temperatura (??K)
  a12: float
    Parametro A12 de wilson
  a21: float
    Parametro A21 de wilson

  Devuelve
  --------
  list[float]
    lambda12 - lambda11, lambda12 - lambda 22
  """

  v1l, v2l = global_vl_cm3_mol

  # Despeje ecuaci??n 5-39 seader
  # lambda12 - lambda11
  l12l11 = -np.log(a12 * v1l / v2l) * global_r_j_molk * t
  # Despeje ecuaci??n 5-40 seader
  # lambda12 - lambda2
  l12l22 = -np.log(a21 * v2l / v1l) * global_r_j_molk * t

  return l12l11, l12l22


def wilson_entalpia_en_exceso(x, t, a12, a21):
  """
  Funcion para calcular la entalpia en exceso con la ecuaci??n 5-55 del seader

  Parametros
  ----------
  x: list[float]
    Lista con los valores de x1 y x2
  t: float
    Temperatura (??K)
  a12: float
    Parametro A12 de wilson
  a21: float
    Parametro A21 de wilson

  Devuelve
  --------
  float
    Entalpia en exceso (J / mol)
  """

  x1, x2 = x
  l12l11, l12l22 = wilson_lambdas(t, a12, a21)

  # Ecuaci??n 5-55
  a = x1 * (x2 * a12 / (x1 + x2 * a12)) * l12l11
  b = x2 * (x1 * a21 / (x2 + x1 * a21)) * l12l22

  return a + b
 

def vanlaar_gamma_1(x, a21, a12):
  """
  Parametros
  ----------
  x: list[float]
    Composici??n de la mezcla, valores de x1 y x2
  a12: float
    Valor A12 de van laar
  a21: float
    Valor A21 de van laar

  Devuelve
  --------
  float
    gamma 1 de van laar
  """

  x1, x2 = x

  a = (a21 * x2 / (a12 * x1 + a21 * x2))

  return np.exp(a12 * a * a)
 
  
def vanlaar_gamma_2(x, a21, a12):
  """
  Parametros
  ----------
  x: list[float]
    Composici??n de la mezcla, valores de x1 y x2
  a12: float
    Valor A12 de van laar
  a21: float
    Valor A21 de van laar

  Devuelve
  --------
  float
    gamma 1 de van laar
  """

  x1, x2 = x

  a = (a12 * x1 / (a12 * x1 + a21 * x2))

  return np.exp(a21 * a * a)



def entalpia_ideal_metanol_liquido(t):
  """
  Funcion para calcular la entalpia ideal del metanol liquido
  tomando como referencia estado liquido y global_t_referencia_kelvin
  como temperatura de referencia

  Parametros
  ----------
  t: float
    Temperatura (??K)

  Devuelve
  ---------
  float
    H^id en (J / mol)
  """

  # Se consigue la entalpia de ref en ??C
  t1 = global_t_referencia_kelvin - 273.15


  # Se integra el polinomio
  entalpia_l = global_cp_metanol_l_kj_molc.integ()


  # Se saca al entalpia en J / mol
  return (entalpia_l(t - 273.15) - entalpia_l(t1)) * 1000


def entalpia_ideal_benceno_liquido(t):
  """
  Funcion para calcular la entalpia ideal del benceno liquido
  tomando como referencia estado liquido y global_t_referencia_kelvin
  como temperatura de referencia

  Parametros
  ----------
  t: float
    Temperatura (??K)

  Devuelve
  ---------
  float
    H^id en (J / mol)
  """

  # Se consigue la entalpia de ref en ??C
  t1 = global_t_referencia_kelvin - 273.15

  # Se integra el polinomio
  entalpia_l = global_cp_benceno_l_kj_molc.integ()

  # Se saca al entalpia en J / mol
  return (entalpia_l(t - 273.15) - entalpia_l(t1)) * 1000


def entalpia_ideal_metanol_ref_a_vap():
  """
  Funcion para calcular entalpia de Tref a estado vapor para el metanol

  Devuelve
  ---------
  float
    H^id en (J / mol)
  """
  
  # Se guardan globales en variables mas peque??a para uso inmediato
  tb = global_tb_metanol_kelvin
  hv = global_hv_metanol_j_mol
  return entalpia_ideal_metanol_liquido(tb) + hv


def entalpia_ideal_benceno_ref_a_vap():
  """
  Funcion para calcular entalpia de Tref a estado vapor para el benceno

  Devuelve
  ---------
  float
    H^id en (J / mol)
  """

  # Se guardan globales en variables mas peque??a para uso inmediato
  tb = global_tb_benceno_kelvin
  hv = global_hv_benceno_j_mol
  return entalpia_ideal_benceno_liquido(tb) + hv


def entalpia_ideal_metanol_vapor(t):
  """
  Funcion para calcular la entalpia ideal del metanol en estado vapor a t

  Parametros
  ----------
  t: float
    Temperatura (??K)

  Devuelve
  ---------
  float
    H^id en (J / mol)
  """
  # Se saca la Tb en celsius
  tb = global_tb_metanol_kelvin - 273.15

  # Se integra el cp
  entalpia_v = global_cp_metanol_v_kj_molc.integ()

  # Se saca la parte del estado de referencia a vapor
  vap = entalpia_ideal_metanol_ref_a_vap()
  return vap + (entalpia_v(t - 273.15) - entalpia_v(tb)) * 1000

  
def entalpia_ideal_benceno_vapor(t):
  """
  Funcion para calcular la entalpia ideal del metanol en estado vapor a t

  Parametros
  ----------
  t: float
    Temperatura (??K)

  Devuelve
  ---------
  float
    H^id en (J / mol)
  """
  
  # Se saca la Tb en celsius
  tb = global_tb_benceno_kelvin - 273.15

  # Se integra el cp
  entalpia_v = global_cp_benceno_v_kj_molc.integ()

  # Se saca la parte del estado de referencia a vapor
  vap = entalpia_ideal_benceno_ref_a_vap()
  return vap + (entalpia_v(t - 273.15) - entalpia_v(tb)) * 1000
  
  
def redlich_kwong_entalpia_reducida(y, p, t):
  """
  Funcion para calcular la entalpia reducida segun la ecuaci??n 4-64 del seader

  Parametros
  ----------
  y: list[float]
    Lista con los valores de y1 y y2
  p: float
    Presi??n (kPa)
  t: float
    Temperatura (??K)

  Devuelve
  ---------
  float
    H^R en (J / mol)
  """

  tc = global_tc_kelvin
  pc = global_pc_kpa

  # Si assert evalua a falso el programa se para
  # Asegurarse de que las longitudes de las listas sean las mismas
  assert len(tc) == len(pc) == len(y)

  # Calcular lista de tr
  tr = [t / tci for tci in tc]

  # calcular lista de Ai y Bi
  lista_ai = []
  lista_bi = []
  for pci, tri in zip(pc, tr):
    lista_ai.append(redlich_kwong_ai(pci, tri))
    lista_bi.append(redlich_kwong_bi(pci, tri))

  # Calcular A y B por reglas de mezclado
  a = regla_de_mezcla_a(lista_ai, y)
  b = regla_de_mezcla_b(lista_bi, y)
  
  z = redlich_kwong_calcular_compresibilidad(y, p, t)
  p_psia = de_kpa_a_psia(p)

  rt = global_r_j_molk * t

  return rt *  (z - 1 - 3 * a * a / (2 * b) * np.log(1 + b * p_psia / z))


def entalpia_ideal_liquido(x, t):
  """
  Funcion para calcula la entalpia ideal liquida de la mezcla

  Parametros
  ----------
  x : list[float]
    Lista con los valores de x1 y x2
  t : float
    Temperatura (??K)

  Devuelve
  --------
  float
    entalpia liquida (J / mol)
  """

  h1 = entalpia_ideal_metanol_liquido(t)
  h2 = entalpia_ideal_benceno_liquido(t)
  x1, x2 = x
  return x1 * h1 + x2 * h2


def entalpia_ideal_vapor(y, t):
  """
  Funcion para calcula la entalpia ideal vapor de la mezcla

  Parametros
  ----------
  y : list[float]
    Lista con los valores de y1 y y2
  t : float
    Temperatura (??K)

  Devuelve
  --------
  float
    entalpia vapor (J / mol)
  """

  h1 = entalpia_ideal_metanol_vapor(t)
  h2 = entalpia_ideal_benceno_vapor(t)
  y1, y2 = y

  return y1 * h1 + y2 * h2


def wilson_entalpia_no_ideal_liquido(x, t, a12, a21):
  """
  Funcion para calcula la entalpia no ideal de un liquido con wilson

  Parametros
  ----------
  x : list[float]
    Lista con los valores de x1 y x2
  t : float
    Temperatura (??K)
  a12 : float
    Valor A12 de wilson
  a21 : float
    Valor A21 de wilson

  Devuelve
  --------
  float
    entalpia liquida (J / mol)
  """

  # Se calcula la entalpai en exceso
  he = wilson_entalpia_en_exceso(x, t, a12, a21)

  # Se le agrega a la ideal
  return entalpia_ideal_liquido(x, t) + he


def redlich_kwong_entalpia_no_ideal_vapor(y, p, t):
  """
  Funcion para calcula la entalpia no ideal de un liquido con wilson

  Parametros
  ----------
  y : list[float]
    Lista con los valores de y1 y y2
  p : float
    Presi??n (kPa)
  t : float
    Temperatura (??K)

  Devuelve
  --------
  float
    entalpia liquida (J / mol)
  """
  # Se calcula la entalpia reducida
  hr = redlich_kwong_entalpia_reducida(y, p, t)

  # Se le suma ala ideal
  return entalpia_ideal_vapor(y, t) + hr


def wilson_gamma_phi_azeotropo(t):
  """
  Funcion para calcular azeotropo con gamma_phi
  
  Parametros
  ----------
  t: float
    Temperatura (??K)

  Devuelve
  list[Union[float, list]]
    P, [x1, x2], [y1, y2]
  """

  a12, a21 = calcular_wilson_a12_a21_gamma_phi(t)

  def funcion_objetivo(x1):
    x = [x1, 1 - x1]
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]
    _, y = bublp_gamma_phi(x, gamma, t)
    return x1 - y[0]

  result = root_scalar(funcion_objetivo, bracket=[0.01, 0.99])

  if result.converged:
    x1 = result.root
    x = [x1, 1 - x1]
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]
    p, y = bublp_gamma_phi(x, gamma, t)
    return p, y, x

  raise RuntimeError("no se llego a un resultado")


def wilson_raoult_mod_azeotropo(t):
  """
  Funcion para calcular azeotropo con raoult modificada
  
  Parametros
  ----------
  t: float
    Temperatura (??K)

  Devuelve
  list[Union[float, list]]
    P, [x1, x2], [y1, y2]
  """

  a12, a21 = calcular_wilson_a12_a21_gamma_phi(t)

  def funcion_objetivo(x1):
    x = [x1, 1 - x1]
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]
    _, y = bublp_raoult_mod(x, gamma, t)
    return x1 - y[0]

  result = root_scalar(funcion_objetivo, bracket=[0.01, 0.99])

  if result.converged:
    x1 = result.root
    x = [x1, 1 - x1]
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]
    p, y = bublp_raoult_mod(x, gamma, t)
    return p, y, x

  raise RuntimeError("no se llego a un resultado")
  
###############################################################################
# Inicio de la ejecuci??n del codigo
###############################################################################

def main():
  plt.figure(global_numero_figura_general)
  plt.plot(global_x1_experimentales, 
           global_p_experimentales, 
           "s", 
           label="punto de burbuja experimental")
  plt.plot(global_y1_experimentales, 
           global_p_experimentales, 
           "v", 
           label="punto de rocio experimental")


  # Se inicalizan los puntos en x utilizados para graficar
  x_graf = np.linspace(0, 1, 128)


  # Raoult

  # Se inicializan las listas para los valores de y y la presi??n calculadas
  # en referencia a las x experimentales
  y_calc = []
  p_calc = []
  
  for x1_actual in global_x1_experimentales:
    # Se inicializa la lista de x
    x = [x1_actual, 1 - x1_actual]

    # Se calcula la presi??n y la y
    p, y = bublp_raoult(x, global_t_kelvin)
    
    # Se agrega la presi??n y la y1 a la lista
    p_calc.append(p)
    y_calc.append(y[0])

  # Se inicializan las listas de y y presi??n para graficar
  y_graf = []
  p_graf = []
  hl_graf = []
  hv_graf = []

  for x1_actual in x_graf:
    # Se inicializa la lista de x
    x = [x1_actual, 1 - x1_actual]

    # Se calcula la presi??n y la y
    p, y = bublp_raoult(x, global_t_kelvin)
    hl = entalpia_ideal_liquido(x, global_t_kelvin)
    hv = entalpia_ideal_vapor(y, global_t_kelvin)

    # Se agrega la presi??n y la y1 a la lista
    p_graf.append(p)
    y_graf.append(y[0])
    hl_graf.append(hl)
    hv_graf.append(hv)

  # Se muestran los resultados
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "Raoult",
                hl_graf,
                hv_graf,
                "x")



  # Raoult Modificada

  # Se calculan los coeficientes de A12 y A21 para wilson
  a12, a21 = calcular_wilson_a12_a21_raoult_mod(global_t_kelvin)

  # Se borran los valores pasados de y y presi??n
  y_calc = []
  p_calc = []


  for x1_actual in global_x1_experimentales:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    
  # Se calculan las gammas
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]
    
    # Se calcula la presi??n y las y con bublp
    p, y = bublp_raoult_mod(x, gamma, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_calc.append(p)
    y_calc.append(y[0])

  # Se borran los valores pasados de y y presi??n para graficar
  y_graf = []
  p_graf = []
  hl_graf = []
  hv_graf = []

  for x1_actual in x_graf:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]

    # Se calcula la presi??n y las y con bublp
    p, y = bublp_raoult_mod(x, gamma, global_t_kelvin)

    hl = wilson_entalpia_no_ideal_liquido(x, global_t_kelvin, a12, a21)
    hv = redlich_kwong_entalpia_no_ideal_vapor(y, p, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_graf.append(p)
    y_graf.append(y[0])
    hl_graf.append(hl)
    hv_graf.append(hv)

  # Se muestran los datos
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "Raoult Modificada",
                hl_graf,
                hv_graf,
                "1")

  # Gamma-Phi

  # Se calculan los coeficientes de A12 y A21 para wilson con gamma-phi
  a12, a21 = calcular_wilson_a12_a21_gamma_phi(global_t_kelvin)


  # Se borran los valores pasados de y y presi??n
  y_calc = []
  p_calc = []

  for x1_actual in global_x1_experimentales:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]


    # Se calcula la presi??n y las y con bublp
    p, y = bublp_gamma_phi(x, gamma, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_calc.append(p)
    y_calc.append(y[0])


  # Se borran los valores pasados de y y presi??n para graficar
  y_graf = []
  p_graf = []
  hl_graf = []
  hv_graf = []

  for x1_actual in x_graf:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]

    # Se calcula la presi??n y las y con bublp
    p, y = bublp_gamma_phi(x, gamma, global_t_kelvin)
    hl = wilson_entalpia_no_ideal_liquido(x, global_t_kelvin, a12, a21)
    hv = redlich_kwong_entalpia_no_ideal_vapor(y, p, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_graf.append(p)
    y_graf.append(y[0])
    hl_graf.append(hl)
    hv_graf.append(hv)

  # Se muestran los datos
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "Gamma-Phi",
                hl_graf,
                hv_graf)


  # Se borran los valores pasados de y y presi??n
  y_calc = []
  p_calc = []

  for x1_actual in global_x1_experimentales:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calcula la presi??n y las y con bublp
    p, y = bublp_flash(x, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_calc.append(p)
    y_calc.append(y[0])


  # Se borran los valores pasados de y y presi??n para graficar
  y_graf = []
  p_graf = []

  for x1_actual in x_graf:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calcula la presi??n y las y con flash
    p, y = bublp_flash(x, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_graf.append(p)
    y_graf.append(y[0])

  # Se muestan los datos
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "Flash")


  """
  # valores de lit para van laar
  a12, a21 = 2.1623, 1.7925

  # Se borran los valores pasados de y y presi??n
  y_calc = []
  p_calc = []

  for x1_actual in global_x1_experimentales:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = vanlaar_gamma_1(x, a12, a21)
    gamma2 = vanlaar_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]

    # Se calcula la presi??n y las y con bublp
    p, y = bublp_raoult_mod(x, gamma, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_calc.append(p)
    y_calc.append(y[0])


  # Se borran los valores pasados de y y presi??n para graficar
  y_graf = []
  p_graf = []

  for x1_actual in x_graf:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = vanlaar_gamma_1(x, a12, a21)
    gamma2 = vanlaar_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]

    # Se calcula la presi??n y las y con bublp
    p, y = bublp_raoult_mod(x, gamma, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_graf.append(p)
    y_graf.append(y[0])

  # Se muestan los datos
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "vanlaar raoult mod")

  # Se borran los valores pasados de y y presi??n
  y_calc = []
  p_calc = []

  for x1_actual in global_x1_experimentales:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = vanlaar_gamma_1(x, a12, a21)
    gamma2 = vanlaar_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]

    # Se calcula la presi??n y las y con bublp
    p, y = bublp_gamma_phi(x, gamma, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_calc.append(p)
    y_calc.append(y[0])


  # Se borran los valores pasados de y y presi??n para graficar
  y_graf = []
  p_graf = []

  for x1_actual in x_graf:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = vanlaar_gamma_1(x, a12, a21)
    gamma2 = vanlaar_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]

    # Se calcula la presi??n y las y con bublp
    p, y = bublp_gamma_phi(x, gamma, global_t_kelvin)

    # Se agregan los valores de presi??n y y a las listas
    p_graf.append(p)
    y_graf.append(y[0])

  # Se muestan los datos
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "vanlaar gamma phi")
  """

  print(wilson_gamma_phi_azeotropo(global_t_kelvin))
  print(wilson_raoult_mod_azeotropo(global_t_kelvin))
  mostrar_figuras()


if __name__ == "__main__":
  main()
