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

from scipy.optimize import minimize


###############################################################################
# Declaración de constantes globales
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
global_vl_cm3_mol = [40.5, 89.4]
global_r_j_molk = 8.314

###############################################################################
# Declaración de metodos
###############################################################################


def de_kpa_a_psia(p):
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
    Coeficiente A de la ecuación de antoine
  b : float
    Coeficiente B de la ecuación de antoine
  c : float
    Coeficiente C de la ecuación de antoine
  t : float
    Temperatura en ˚C

  Devuelve
  --------
  float
    presión saturada en kpa
  """
  return np.exp(a - b / (t + c))


def antoine_smith_metanol_presion_saturada(t):
  """
  Parametros
  ----------
  t : float
    Temperatura en ˚K

  Devuelve
  --------
  float
    Presión saturada del metanol en kPa
  """
  return antoine_smith_presion_saturada(16.5785, 3638.27, 239.500, t - 273.15)


def antoine_smith_benceno_presion_saturada(t):
  """
  Parametros
  ----------
  t : float
    Temperatura en ˚K

  Devuelve
  --------
  float
    Presión saturada del benceno en kPa
  """
  return antoine_smith_presion_saturada(13.7819, 2726.81, 217.572, t - 273.15)


def redlich_kwong_equacion_cubica(a, b, p, z):
  """
  Funcion que representa la ecuacíon 4-41 del seader

  Z^3 - Z^2 + BP(A^2/B - BP - 1)Z - A^2/B(BP)^2 = f(Z)

  Parametros
  ----------
  a : float
    Regla de mezcla A, descrita por la ecuación 4-42 del seader
  b : float
    Reglas de mezcla B, descrita por la ecuaíon 4-43 del seader
  p : float
    Presión
  z : float (kPa)
    Factor de compresibilidad

  Devuelve
  --------
  float
    El resultado de la función f(Z)
  """

  # Se recalcula la p como psia
  p_psia = de_kpa_a_psia(p)

  # Precalcular variables para representar la ecuación como:
  # Z^3 - Z^2 + a1 * Z + a0 = f(Z)

  a0 = -a * a / b * (b * p_psia) * (b * p_psia)
  a1 = b * p_psia * (a * a / b - b * p_psia - 1)
  return z * z * z - z * z + a1 * z + a0


def redlich_kwong_equacion_cubica_derivada(a, b, p, z):
  """
  metodo que representa al derivada de la ecuación 4-41 del seader

  f'(Z) = 3Z^2 - 2Z + BP(((A^2) / B) - BP - 1)

  Parametros
  ----------
  a : float
    Regla de mezcla A, descrita por la ecuación 4-42 del seader
  b : float
    Reglas de mezcla B, descrita por la ecuaíon 4-43 del seader
  p : float
    Presión (kPa)
  z : float
    Factor de compresibilidad

  Devuelve
  --------
  float
    El resultado de la función f'(Z)
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
    Regla de mezcla A, descrita por la ecuación 4-42 del seader
  b : float
    Reglas de mezcla B, descrita por la ecuaíon 4-43 del seader
  p : float
    Presión (kPa)
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

  # Primera iteración de newton rapson para comenzar el ciclo
  z_actual = z_pasada - redlich_kwong_equacion_cubica(a, b, p, z_pasada) /\
      redlich_kwong_equacion_cubica_derivada(a, b, p, z_pasada)

  while abs((z_actual - z_pasada) / z_actual * 100) > err:
    # Se guarda el valor de z en z_pasada
    z_pasada = z_actual

    # Se calcula la siguiente iteración de newton rapson
    z_actual = z_pasada - redlich_kwong_equacion_cubica(a, b, p, z_pasada) /\
        redlich_kwong_equacion_cubica_derivada(a, b, p, z_pasada)

  return z_actual


def regla_de_mezcla_a(a, y):
  """
  Regla de mezclado del parametro A descrita por la ecuación 4-42 del seader

  A = ΣAixi 

  Parametros
  ----------
  A : list[float]
    Lista con los valores de A en cada posición de i
  y : list[float]
    Lista con los valores de la composición de la mezcla

  Devuelve
  --------
  float
    Parametro A de la ecuación Redlich-Kwong
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
  Regla de mezclado del parametro B descrita por la ecuación 4-43 del seader

  B = ΣBiyi 

  Parametros
  ----------
  b : list[float]
    Lista con los valores de B en cada posición de i
  x: list[float]
    Lista con los valores de la composición de la mezcla

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
  Valor de A para la substancia i segun la ecuación 4-44 del seader

  Parametros
  ----------
  pc : float
    Presión critica (kPa)

  tr : float
    Temperatura reducida  (˚K)

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
  Valor de A para la substancia i segun la ecuación 4-44 del seader

  Parametros
  ----------
  pc : float
    Presión critica (kPa)

  tr : float
    Temperatura reducida  (˚K)

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
  Funcion para calcular la fugacidad descrita de la ecuación 4-72 del seader

  φi = e^[(Z - 1)Bi/B - ln(Z - BP) - A^2/B(2Ai/A - Bi/B)ln(1 + BP/Z)]

  Parametros
  ----------
  z : float
    Factor de compresiblidad
  a : float
    Regla de mezcla A, descrita por la ecuación 4-42 del seader
  ai : float
    Valor de A para la substancia i
  b : float
    Reglas de mezcla B, descrita por la ecuaíon 4-43 del seader
  bi : float
    Valor de B para la substancia i
  p : float
    Presión (kPa)

  Devuelve
  --------
  float
    Fugacidad de la substancia i liquida
  """

  # El seader toma la p como psias 
  p_psia = de_kpa_a_psia(p)

  # La funcion se va a separar para simplificarla de la siguiente manera:
  # φiL = e^[(Z - 1)Bi/B - ln(Z - BP) - A^2/B(2Ai/A - Bi/B)ln(1 + BP/Z)]
  # d = (Z - 1)Bi/B
  # e = -ln(Z - BP)
  # f = -A^2/B
  # g = 2Ai/A - Bi/B
  # h = ln(1 + BP/Z)
  # φiL = e^(d + e + f * g * h)

  # Precalculando valores para simplificar
  d = (z - 1) * bi / b
  e = -np.log(z - b * p_psia)
  f = -a * a / b
  g = 2 * ai / a - bi / b
  h = np.log(1 + b * p_psia / z)
  
  return np.exp(d + e + f * g * h)


def redlich_kwong_coeficiente_de_fugacidad_substancia_i(pc, tc, y, p, t, i):
  """
  Funcion para calcular la fugacidad de una substancia i en una mezcla

  Parametros
  ----------
  pc : list[float]
    Una lista con las presiónes criticas para i (kPa)
  tc : list[float]
    Una lista con las temperatura criticas para i (˚K)
  y  : list[float]
    Una list con la composición de la mezcla
  p : float
    La presión (kPa)
  t : float
    La temperatura (˚K)
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
    Presión (kPa)
  t : float
    Temperatura (˚K)

  Devuelve
  --------
  float
    El coeficiente de fugacidad del metanol
  """

  # Se guardan los valores criticos en variables con nombres mas pequeños
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
    Presión (kPa)
  t : float
    Temperatura (˚K)

  Devuelve
  --------
  float
    El coeficiente de fugacidad del benceno
  """

  # Se guardan los valores criticos en variables con nombres mas pequeños
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
    Presión (kPa)
  t : float
    Temperatura (˚K)

  Devuelve
  --------
  float
    El coeficiente de fugacidad del benceno
  """
  
  # Se calcula el coeficiente de fugacidad
  phi = redlich_kwong_fugacidad_metanol(y, p, t)

  # Se calcula la presión saturada del metanol
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
    Presión (kPa)
  t : float
    Temperatura (˚K)

  Devuelve
  --------
  float
    El coeficiente de fugacidad del benceno
  """

  # Se calcula el coeficiente de fugacidad
  phi = redlich_kwong_fugacidad_benceno(y, p, t)

  # Se calcula la presión saturada del benceno
  p_sat = antoine_smith_benceno_presion_saturada(t)

  # Se calcula la phi saturada
  phi_sat = redlich_kwong_fugacidad_benceno(y, p_sat, t)

  # NOTA: se omitio el factor de poynting
  return phi / phi_sat


def wilson_gamma_1(x, a12, a21):
  """
  Ecuación 5-28 del seader para calcular la gamma 1 de wilson

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

  # Se separa la ecuación de la siguiente manera
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
  Ecuación 5-28 del seader para calcular la gamma 2 de wilson

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

  # Se separa la ecuación de la siguiente manera
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
    Temperatura (˚K)

  Devuelve
  -------
  list[Union[float, list]]
    El primer elemento de la lista es la presión, el segundo elemento es una
    lista con los valores de y1 y y2 
  """
  
  # Se inicializa x1 y x2 
  x1, x2 = x

  # Se calculan las presiones saturadas
  p_sat_1 = antoine_smith_metanol_presion_saturada(t)
  p_sat_2 = antoine_smith_benceno_presion_saturada(t)

  # Se calcula la presión
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
    Temperatura (˚K)

  Devuelve
  -------
  list[Union[float, list]]
    El primer elemento de la lista es la presión, el segundo elemento es una
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
  Funcion para calcular la presión segun la ecuación 14.10 del smith

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
    Valores de {φi...} precalculados para la presión y temperatura correcta

  Devuelve
  --------
  float
    Presión (unidades de p_sat)
  """
  
  # Asegurarse que las listas tengan el mismo tamaño
  assert len(x) == len(gamma) == len(p_sat) == len(phi)

  # Se inicializa la presión
  p = 0

  # Se aplica la sumatoria
  for x_i, gamma_i, p_sat_i, phi_i in zip(x, gamma, p_sat, phi):
    p += x_i * gamma_i * p_sat_i / phi_i

  return p


def gamma_phi_y(x, gamma, p_sat, phi, p):
  """
  Funcion para calcular los valores de y segun la ecuación 14.8 del smith

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
    Valores de {φi...} precalculados para la presión y temperatura correcta
  p : float
    Presión (kPa)

  Devuelve
  --------
  list[float]
    Una lista con los valores de y
  """
  # Asegurarse que las listas tengan el mismo tamaño
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
    Temperatura (˚K)
  err : float
    Error porcentual permitido en el calculo de bublp

  Devuelve
  -------
  list[Union[float, list]]
    El primer elemento de la lista es la presión, el segundo elemento es una
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
  
  # Se calcula la P segun la ecuacíon 14.10 del smith
  p = gamma_phi_p(x, gamma, p_sat, phi)

  # Se inicializa una lista de y afuera del scope de el loop
  y = []

  # Se compara la presión pasada y actual
  while abs((p - p_pasada) / p * 100) > err:
    # Se asigna la presión pasada para calcular la nueva
    p_pasada = p

    # Se calculan los valores de y segun la ecuación 14.8 del smith
    y = gamma_phi_y(x, gamma, p_sat, phi, p)

    # Se calculan las phi de metanol y benceno
    phi[0] = redlich_kwong_phi_mayus_metanol(y, p, t)
    phi[1] = redlich_kwong_phi_mayus_benceno(y, p, t)

    # Se saca el nuevo valor de presión
    p = gamma_phi_p(x, gamma, p_sat, phi)

  return p, y


def calcular_wilson_a12_a21_gamma_phi(t):
  """
  Funcion para calcular los valores de A12 y A21 de wilson utilizando
  los valores globales experimentales y bublp de gamma phi


  Utiliza una funcion objetivo
  Fobj = Σ(p_experimental - p_calculada) y la funcion minimizar para 
  encontrar A12 y A21, donde p_calculada se calcula usando bublp_gamma_phi

  Parametros
  ----------
  t : float
    Temperatura (˚K)

  Devuelve
  --------
  list[float]
    Una lista con los valores de A12 y A21
  """
  
  # Se inicializa una lista con los valores de x1 experimentales omitiendo el
  # primer valor de x1 y el ultimo
  x1 = global_x1_experimentales[1:-1]

  # Se inicializa una lista con los valores de presión experimentales 
  # omitiendo el primer valor de presión y el ultimo
  p = global_p_experimentales[1:-1]

  # Se comienza la declaración de la funcion objetivo
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

      # Se calcula la presión con bublp
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

  # Si no se levanta una excepción
  raise RuntimeError("calcular_wilson_a12_a21_gamma_phi: no se llego a un "
                     "resultado")


def calcular_wilson_a12_a21_raoult_mod(t):
  """
  Funcion para calcular los valores de A12 y A21 de wilson utilizando
  los valores globales experimentales y bublp de raoult modificada


  Utiliza una funcion objetivo
  Fobj = Σ(p_experimental - p_calculada) y la funcion minimizar para 
  encontrar A12 y A21, donde p_calculada se calcula usando bublp_raoult_mod

  Parametros
  ----------
  t : float
    Temperatura (˚K)

  Devuelve
  --------
  list[float]
    Una lista con los valores de A12 y A21
  """
  # Se inicializa una lista con los valores de x1 experimentales omitiendo el
  # primer valor de x1 y el ultimo
  x1 = global_x1_experimentales[1:-1]

  # Se inicializa una lista con los valores de presión experimentales 
  # omitiendo el primer valor de presión y el ultimo
  p = global_p_experimentales[1:-1]

  # Se comienza la declaración de la funcion objetivo
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

      # Se calcula la presión con bublp
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

  # Si no se levanta una excepción
  raise RuntimeError("calcular_wilson_a12_a21_gamma_phi: no se llego a un "
                     "resultado")


def mostrar_datos(y_calc, p_calc, x_graf, y_graf, p_graf, titulo, he=None):
  """
  Funcion para mostrar los calculados (faltan energias)

  Parametros
  ----------
  y_calc : list[float]
    Valores de y calculados en referencia a las x en global_x1_experimental
  p_calc : list[float]
    Valores de presión calculados en referencia con las presiones en
    global_p_experimental (kPa)
  x_graf : list[float]
    Lista con los valores de x para graficar
  y_graf : list[float]
    Lista con los valores de y para graficar
  p_graf : list[float]
    Lista con los valores de presión para graficar
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
  plt.grid()
  plt.legend()

  # Se cambia la cuenta de figuras
  global_cuenta_de_figuras += 1

  
  # Si hay entalpias en exceso que graficar
  if he is not None:
    # Se crea una nueva figura
    plt.figure(global_cuenta_de_figuras)

    # Se el asigna el titulo del metodo
    plt.title(titulo)

    # Se grafica
    plt.plot(x_graf, he, label="entalpia en exceso")

    # Se muestra la grafica
    plt.legend()
    plt.grid()
    plt.xlabel("$x_1$")
    plt.ylabel("$H^E$")

    # Se cambia la cuenta de figuras
    global_cuenta_de_figuras += 1

def mostrar_figuras():
  plt.show()


def flash_k(p, t):
  """
  Funcion para calcular los valores de K para flash
  
  Parametros
  ----------
  p: float
    Presión (kPa)
  t: float
    Temperatura (˚K)

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
    Temperatura (˚K)
 
  Devuelve
  --------
  list[Union[float, list]]
    Presión (kPa), [y1, y2]
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

  # La función objetivo descrita por el seader 7-18
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
    y_actual = normalizar_lista(y_actual) 
    
    # Se checa la funcion objetivo
    if funcion_objetivo(z, k) < 1e-15 and funcion_objetivo(z, k) > -1e-15:
      break
  
    # Se vuelve a estimar la presión
    p = sum([zi * p_sat_i for zi, p_sat_i in zip(z, p_sat)])

  return p, y_actual
 

def wilson_lambdas(t, a12, a21):
  """
  Funcion para calcular las diferencias de lambdas para wilson
  para la mezcal metanol - benceno con las ecuaciones 5-39 y 5-40 del seader

  Parametros
  ----------
  t: float
    Temperatura (˚K)
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

  # Despeje ecuación 5-39 seader
  # lambda12 - lambda11
  l12l11 = -np.log(a12 * v1l / v2l) * global_r_j_molk * t
  # Despeje ecuación 5-40 seader
  # lambda12 - lambda2
  l12l22 = -np.log(a21 * v2l / v1l) * global_r_j_molk * t

  return l12l11, l12l22


def wilson_entalpia_en_exceso(x, t, a12, a21):
  """
  Funcion para calcular la entalpia en exceso con la ecuación 5-55 del seader

  Parametros
  ----------
  x: list[float]
    Lista con los valores de x1 y x2
  t: float
    Temperatura (˚K)
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

  # Ecuación 5-55
  a = x1 * (x2 * a12 / (x1 + x2 * a12)) * l12l11
  b = x2 * (x1 * a21 / (x2 + x1 * a21)) * l12l22

  return a + b
  
  
  
###############################################################################
# Inicio de la ejecución del codigo
###############################################################################

def main():
  # NOTE(samuel): este codigo se repite mucho, pasarlo a una funcion que 
  # acepte la funcion bublp que se vaya a utilizar

  # Se inicalizan los puntos en x utilizados para graficar
  x_graf = np.linspace(0, 1, 128)


  # Raoult

  # Se inicializan las listas para los valores de y y la presión calculadas
  # en referencia a las x experimentales
  y_calc = []
  p_calc = []

  for x1_actual in global_x1_experimentales:
    # Se inicializa la lista de x
    x = [x1_actual, 1 - x1_actual]

    # Se calcula la presión y la y
    p, y = bublp_raoult(x, global_t_kelvin)

    # Se agrega la presión y la y1 a la lista
    p_calc.append(p)
    y_calc.append(y[0])

  # Se inicializan las listas de y y presión para graficar
  y_graf = []
  p_graf = []

  for x1_actual in x_graf:
    # Se inicializa la lista de x
    x = [x1_actual, 1 - x1_actual]

    # Se calcula la presión y la y
    p, y = bublp_raoult(x, global_t_kelvin)

    # Se agrega la presión y la y1 a la lista
    p_graf.append(p)
    y_graf.append(y[0])

  # Se muestran los resultados
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "raoult")

  # Raoult Modificada

  # Se calculan los coeficientes de A12 y A21 para wilson
  a12, a21 = calcular_wilson_a12_a21_raoult_mod(global_t_kelvin)

  # Se borran los valores pasados de y y presión
  y_calc = []
  p_calc = []


  for x1_actual in global_x1_experimentales:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    
  # Se calculan las gammas
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]
    
    # Se calcula la presión y las y con bublp
    p, y = bublp_raoult_mod(x, gamma, global_t_kelvin)

    # Se agregan los valores de presión y y a las listas
    p_calc.append(p)
    y_calc.append(y[0])

  # Se borran los valores pasados de y y presión para graficar
  y_graf = []
  p_graf = []
  he_graf = []

  for x1_actual in x_graf:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]

    # Se calcula la presión y las y con bublp
    p, y = bublp_raoult_mod(x, gamma, global_t_kelvin)
    he = wilson_entalpia_en_exceso(x, global_t_kelvin, a12, a21)

    # Se agregan los valores de presión y y a las listas
    p_graf.append(p)
    y_graf.append(y[0])
    he_graf.append(he)

  # Se muestran los datos
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "raoult mod",
                he_graf)


  # Gamma-Phi

  # Se calculan los coeficientes de A12 y A21 para wilson con gamma-phi
  a12, a21 = calcular_wilson_a12_a21_gamma_phi(global_t_kelvin)


  # Se borran los valores pasados de y y presión
  y_calc = []
  p_calc = []

  for x1_actual in global_x1_experimentales:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]


    # Se calcula la presión y las y con bublp
    p, y = bublp_gamma_phi(x, gamma, global_t_kelvin)

    # Se agregan los valores de presión y y a las listas
    p_calc.append(p)
    y_calc.append(y[0])


  # Se borran los valores pasados de y y presión para graficar
  y_graf = []
  p_graf = []
  he_graf = []

  for x1_actual in x_graf:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calculan las gammas
    gamma1 = wilson_gamma_1(x, a12, a21)
    gamma2 = wilson_gamma_2(x, a12, a21)
    gamma = [gamma1, gamma2]

    # Se calcula la presión y las y con bublp
    p, y = bublp_gamma_phi(x, gamma, global_t_kelvin)
    he = wilson_entalpia_en_exceso(x, global_t_kelvin, a12, a21)

    # Se agregan los valores de presión y y a las listas
    p_graf.append(p)
    y_graf.append(y[0])
    he_graf.append(he)

  # Se muestan los datos
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "gamma-phi",
                he_graf)

  # Se borran los valores pasados de y y presión
  y_calc = []
  p_calc = []

  for x1_actual in global_x1_experimentales:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calcula la presión y las y con bublp
    p, y = bublp_flash(x, global_t_kelvin)

    # Se agregan los valores de presión y y a las listas
    p_calc.append(p)
    y_calc.append(y[0])


  # Se borran los valores pasados de y y presión para graficar
  y_graf = []
  p_graf = []

  for x1_actual in x_graf:
    # Se inicializan los valores de x
    x = [x1_actual, 1 - x1_actual]

    # Se calcula la presión y las y con flash
    p, y = bublp_flash(x, global_t_kelvin)

    # Se agregan los valores de presión y y a las listas
    p_graf.append(p)
    y_graf.append(y[0])

  # Se muestan los datos
  mostrar_datos(y_calc,
                p_calc,
                x_graf,
                y_graf,
                p_graf,
                "flash")

  mostrar_figuras()


if __name__ == "__main__":
  main()
