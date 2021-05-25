###############################################################################
#  ________    ___      ________          
# |\   __  \  |\  \    |\   __  \         
# \ \  \|\  \ \ \  \   \ \  \|\  \        
#  \ \   ____\ \ \  \   \ \   __  \       
#   \ \  \___|  \ \  \   \ \  \ \  \      
#    \ \__\      \ \__\   \ \__\ \__\     
#     \|__|       \|__|    \|__|\|__|     
#
#  _________    _______       ________      _____ ______       ________     
# |\___   ___\ |\  ___ \     |\   __  \    |\   _ \  _   \    |\   __  \    
# \|___ \  \_| \ \   __/|    \ \  \|\  \   \ \  \\\__\ \  \   \ \  \|\  \   
#      \ \  \   \ \  \_|/__   \ \   _  _\   \ \  \\|__| \  \   \ \  \\\  \  
#       \ \  \   \ \  \_|\ \   \ \  \\  \|   \ \  \    \ \  \   \ \  \\\  \ 
#        \ \__\   \ \_______\   \ \__\\ _\    \ \__\    \ \__\   \ \_______\                                                                                                                                                                                                                                                                                                                                                                                                             
#         \|__|    \|_______|    \|__|\|__|    \|__|     \|__|    \|_______|
# ________      _______       ___            _______       ________      
# |\   ___ \    |\  ___ \     |\  \          |\  ___ \     |\   __  \     
# \ \  \_|\ \   \ \   __/|    \ \  \         \ \   __/|    \ \  \|\  \    
# \ \  \ \\ \   \ \  \_|/__   \ \  \         \ \  \_|/__   \ \  \\\  \   
#  \ \  \_\\ \   \ \  \_|\ \   \ \  \____     \ \  \_|\ \   \ \  \\\  \  
#   \ \_______\   \ \_______\   \ \_______\    \ \_______\   \ \_____  \ 
#    \|_______|    \|_______|    \|_______|     \|_______|    \|___| \__\
#
###############################################################################
# Librerias utilizadas
###############################################################################


import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize, minimize_scalar


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
global_vl_cm3_mol = [40.5, 89.4]
global_delta_cal_cm3 = [14.510, 9.158]
global_r_cal_molk = 1.9872
global_n = 0.2


###############################################################################
# Declaración de metodos
###############################################################################


def rmsep(v_observado, v_experimental):
  assert len(v_observado) == len(v_experimental)

  n = len(v_observado)

  s = 0
  for v_obs, v_exp in zip(v_observado, v_experimental):
    diff = v_obs - v_exp
    s += diff * diff

  return np.sqrt(s / n)


def antoine_smith_presion_saturada(A, B, C, T):
  """
  Parametros
  ----------
  A : float
    Coeficiente A de la ecuación de antoine
  B : float
    Coeficiente B de la ecuación de antoine
  C : float
    Coeficiente C de la ecuación de antoine
  T : float
    Temperatura en ˚C

  Devuelve
  float
    Presión saturada en kPa
  """
  return np.exp(A - B / (T + C))


def antoine_smith_metanol_presion_saturada(T):
  """
  Parametros
  ----------
  T : float
    Temperatura en ˚K

  Devuelve
  float
    Presión saturada del metanol en kPa
  """
  return antoine_smith_presion_saturada(16.5785, 3638.27, 239.500, T - 273.15)


def antoine_smith_benceno_presion_saturada(T):
  """
  Parametros
  ----------
  T : float
    Temperatura en ˚K

  Devuelve
  float
    Presión saturada del benceno en kPa
  """
  return antoine_smith_presion_saturada(13.7819, 2726.81, 217.572, T - 273.15)


def redlich_kwong_alpha(Tr):
  """
  Valor de alpha para la ecuación de Redlich-Kwong

  Parametros
  ----------
  Tr : float
    Temperatura reducida de la substancia

  Devuelve
  -------
  float
    Valor de alpha
  """

  return np.sqrt(Tr)


def redlich_kwong_equacion_cubica(A, B, P, Z):
  """
  Metodo que representa la ecuacíon 4-41 del seader

  Z^3 - Z^2 + BP(A^2/B - BP - 1)Z - A^2/B(BP)^2 = f(Z)

  Parametros
  ----------
  A : float
    Regla de mezcla A, descrita por la ecuación 4-42 del seader
  B : float
    Reglas de mezcla B, descrita por la ecuaíon 4-43 del seader
  P : float
    Presión
  Z : float
    Factor de compresibilidad

  Devuelve
  --------
  float
    El resultado de la función f(Z)
  """

  # Precalcular variables para representar la ecuación como:
  # Z^3 - Z^2 + a1 * Z + a0 = f(Z)
  a0 = -A * A / B * (B * P) * (B * P)
  a1 = B * P * (A * A / B - B * P - 1)
  return Z * Z * Z - Z * Z + a1 * Z + a0


def redlich_kwong_equacion_cubica_derivada(A, B, P, Z):
  """
  Metodo que representa al derivada de la ecuación 4-41 del seader

  f'(Z) = 3Z^2 - 2Z + BP(((A^2) / B) - BP - 1)

  Parametros
  ----------
  A : float
    Regla de mezcla A, descrita por la ecuación 4-42 del seader
  B : float
    Reglas de mezcla B, descrita por la ecuaíon 4-43 del seader
  P : float
    Presión
  Z : float
    Factor de compresibilidad

  Devuelve
  --------
  float
    El resultado de la función f'(Z)
  """

  C = B * P * ((A * A) / B - B * P - 1)
  return 3 * Z * Z - 2 * Z + C


def redlich_kwong_resolver_equacion_cubica(A, B, P, Z0=1, err=5e-10):
  """
  Metodo para encontrar el valor de Z

  Parametros
  ----------
  A : float
    Regla de mezcla A, descrita por la ecuación 4-42 del seader
  B : float
    Reglas de mezcla B, descrita por la ecuaíon 4-43 del seader
  P : float
    Presión
  Z0 : float
    Si se quiere encontrar el valor de Z liquido, Z0 tiene que ser cercano a 0
    Si se quiere encontrar el valor de Z vapor, Z0 tiene que ser cercano a 1
  err : float
    El error relativo

  Devuelve
  --------
  float
    El valor de Z encontrado
  """

  Z_pasada = Z0

  # Se va a utilizar newton-rapson
  # Z_i+1 = Z_i - f(Z_i) / f'(Z_i)
  Z_actual = Z_pasada - redlich_kwong_equacion_cubica(A, B, P, Z_pasada) /\
      redlich_kwong_equacion_cubica_derivada(A, B, P, Z_pasada)

  while abs((Z_actual - Z_pasada) / Z_actual * 100) > err:
    Z_pasada = Z_actual
    Z_actual = Z_pasada - redlich_kwong_equacion_cubica(A, B, P, Z_pasada) /\
        redlich_kwong_equacion_cubica_derivada(A, B, P, Z_pasada)

  return Z_actual


def regla_de_mezcla_A(A, y):
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

  # NOTE:
  # si lo que evalua assert es falso el programa se para
  # len(lista) es la longitud de la list

  # Asegurarse que la lista de A y x tengan la misma longitud
  assert len(A) == len(y)

  # Valor inicial de la sumatoria
  s = 0

  # Sumar los productos
  for Ai, yi in zip(A, y):
    s += Ai * yi

  return s


def regla_de_mezcla_B(B, y):
  """
  Regla de mezclado del parametro B descrita por la ecuación 4-43 del seader

  B = ΣBiyi 

  Parametros
  ----------
  B : list[float]
    Lista con los valores de B en cada posición de i
  xy: list[float]
    Lista con los valores de la composición de la mezcla

  Devuelve
  --------
  float
    Parametro B para la ec de redlich kwong del factor de compresibilidad
  """

  # NOTE:
  # si lo que evalua assert es falso el programa se para
  # len(lista) es la longitud de la list

  # Asegurarse que la lista de B y x tengan la misma longitud
  assert len(B) == len(y)

  # Valor inicial de la sumatoria
  s = 0

  # Sumar los productos
  for Bi, yi in zip(B, y):
    s += Bi * yi

  return s


def redlich_kwong_Ai(Pc, Tr):
  """
  Valor de A para la substancia i segun la ecuación 4-44 del seader

  Parametros
  ----------
  Pc : float
    Presión critica

  Tr : float
    Temperatura reducida 

  Devuelve
  --------
  float
    Valor de Ai
  """

  return np.sqrt(0.4278 / (Pc * np.power(Tr, 2.5)))


def redlich_kwong_Bi(Pc, Tr):
  """
  Valor de A para la substancia i segun la ecuación 4-44 del seader

  Parametros
  ----------
  Pc : float
    Presión critica

  Tr : float
    Temperatura reducida 

  Devuelve
  --------
  float
    Valor de Ai
  """

  return 0.0867 / (Pc * Tr)


def coeficiente_de_fugacidad_para_ecuacion_cubica(Z, A, Ai, B, Bi, P):
  """
  Metodo para calcular la fugacidad descrita de la ecuación 4-72 del seader

  φiL = e^[(Z - 1)Bi/B - ln(Z - BP) - A^2/B(2Ai/A - Bi/B)ln(1 + BP/Z)]

  Parametros
  ----------
  Z : float
    Factor de compresiblidad
  A : float
    Regla de mezcla A, descrita por la ecuación 4-42 del seader
  Ai : float
    Valor de A para la substancia i
  B : float
    Reglas de mezcla B, descrita por la ecuaíon 4-43 del seader
  Bi : float
    Valor de B para la substancia i
  P : float
    Presión

  Devuelve
  --------
  float
    Fugacidad de la substancia i liquida
  """
  # φiL = e^[(Z - 1)Bi/B - ln(Z - BP) - A^2/B(2Ai/A - Bi/B)ln(1 + BP/Z)]
  # a = (Z - 1)Bi/B
  # b = -ln(Z - BP)
  # c = -A^2/B
  # d = 2Ai/A - Bi/B
  # g = ln(1 + BP/Z)
  # φiL = e^[a + b + c * d * g]

  # Precalculando valores para simplificar
  a = (Z - 1) * Bi / B
  b = -np.log(Z - B * P)
  c = -A * A / B
  d = 2 * Ai / A - Bi / B
  g = np.log(1 + B * P / Z)

  return np.exp(a + b + c * d * g)


def redlich_kwong_coeficiente_de_fugacidad_substancia_i(Pc, Tc, y, P, T, i):
  """
  Metodo para calcular la fugacidad de una substancia i en una mezcla

  Parametros
  ----------
  Pc : list[float]
    Una lista con las presiónes criticas para i
  Tc : list[float]
    Una lista con las temperatura criticas para i
  y  : list[float]
    Una list con la composición de la mezcla
  P : float
    La presión
  T : float
    La temperatura
  i : int
    Indice de la substancia comenzando en 1

  Devuelve
  --------
  float
    La fugacidad de la substancia i
  """

  # NOTE:
  # si lo que evalua assert es falso el programa se para
  # len(lista) es la longitud de la list

  # Asegurarse de que las longitudes de las listas sean las mismas
  assert len(Tc) == len(Pc) == len(y)

  # Calcular lista de Tr
  Tr = []
  for Tci in Tc:
    Tr.append(T / Tci)

  # Calcular lista de Ai y bi
  lista_Ai = []
  lista_Bi = []
  for Pci, Tri in zip(Pc, Tr):
    lista_Ai.append(redlich_kwong_Ai(Pci, Tri))
    lista_Bi.append(redlich_kwong_Bi(Pci, Tri))

  # Calcular A y B por reglas de mezclado
  A = regla_de_mezcla_A(lista_Ai, y)
  B = regla_de_mezcla_B(lista_Bi, y)

  # Calcular la Z liquida
  Z = redlich_kwong_resolver_equacion_cubica(A, B, P)

  # Initializar el valor de A y B de la substancia i
  # Las listas en python comienzar a contar desde 0
  Ai = lista_Ai[i - 1]
  Bi = lista_Bi[i - 1]

  return coeficiente_de_fugacidad_para_ecuacion_cubica(Z, A, Ai, B, Bi, P)


def redlich_kwong_fugacidad_metanol(y, P, T):
  Pc = global_pc_kpa
  Tc = global_tc_kelvin
  return redlich_kwong_coeficiente_de_fugacidad_substancia_i(Pc, Tc, y, 
                                                             P, T, 1)


def redlich_kwong_fugacidad_benceno(y, P, T):
  Pc = global_pc_kpa
  Tc = global_tc_kelvin
  return redlich_kwong_coeficiente_de_fugacidad_substancia_i(Pc, Tc, y, 
                                                             P, T, 2)


def redlich_kwong_phi_mayus_metanol(y, P, T):
  phi = redlich_kwong_fugacidad_metanol(y, P, T)
  p_sat = antoine_smith_metanol_presion_saturada(T)
  phi_sat = redlich_kwong_fugacidad_metanol(y, p_sat, T)
  return phi / phi_sat


def redlich_kwong_phi_mayus_benceno(y, P, T):
  phi = redlich_kwong_fugacidad_benceno(y, P, T)
  p_sat = antoine_smith_benceno_presion_saturada(T)
  phi_sat = redlich_kwong_fugacidad_benceno(y, p_sat, T)
  return phi / phi_sat


def wilson_gamma_1(x1, A12, A21):
  """
  Ecuación 5-28 del seader para calcular gamma 1

  Parametros
  ---------
  """
  x2 = 1 - x1
  A = x1 + x2 * A12
  B = A12 / (x1 + x2 * A12)
  C = A21 / (x2 + x1 * A21)
  return np.exp(-np.log(A) + x2 * (B - C))


def wilson_gamma_2(x2, A12, A21):
  """
  """
  x1 = 1 - x2
  A = x2 + x1 * A21
  B = A12 / (x1 + x2 * A12)
  C = A21 / (x2 + x1 * A21)
  return np.exp(-np.log(A) - x1 * (B - C))


def vanlaar_gamma_1(x1, A12, A21):
  x2 = 1 - x1
  den = 1 + A12 * x1 / (A21 * x2)
  return np.exp(A12 / (den * den))


def vanlaar_gamma_2(x2, A12, A21):
  x1 = 1 - x2
  den = 1 + A21 * x2 / (A12 * x1)
  return np.exp(A21 / (den * den))


def bublp_raoult(x, T):
  x1, x2 = x
  p_sat_1 = antoine_smith_metanol_presion_saturada(T)
  p_sat_2 = antoine_smith_benceno_presion_saturada(T)

  p = x1 * p_sat_1 + x2 * p_sat_2
  y1 = x1 * p_sat_1 / p
  y2 = x2 * p_sat_2 / p

  return p, [y1, y2]


def bublp_raoult_mod(x, gamma, T):
  x1, x2 = x
  gamma1, gamma2 = gamma

  p_sat_1 = antoine_smith_metanol_presion_saturada(T)
  p_sat_2 = antoine_smith_benceno_presion_saturada(T)

  p = x1 * gamma1 * p_sat_1 + x2 * gamma2 * p_sat_2
  y1 = x1 * gamma1 * p_sat_1 / p
  y2 = x2 * gamma2 * p_sat_2 / p

  return p, [y1, y2]


def gamma_phi_p(x, gamma, p_sat, phi):
  p = 0

  for x_i, gamma_i, p_sat_i, phi_i in zip(x, gamma, p_sat, phi):
    p += x_i * gamma_i * p_sat_i / phi_i

  return p


def gamma_phi_y(x, gamma, p_sat, phi, P):
  y = []

  for x_i, gamma_i, p_sat_i, phi_i in zip(x, gamma, p_sat, phi):
    y.append(x_i * gamma_i * p_sat_i / (phi_i * P))

  return y


def bublp_phi_gamma(x, gamma, T, err=5e-2):
  phi = [1.0, 1.0]

  p_sat_1 = antoine_smith_metanol_presion_saturada(T)
  p_sat_2 = antoine_smith_benceno_presion_saturada(T)
  p_sat = [p_sat_1, p_sat_2]

  p_pasada = 0
  p = gamma_phi_p(x, gamma, p_sat, phi)

  y = []

  while abs((p - p_pasada) / p * 100) > err:
    p_pasada = p
    y = gamma_phi_y(x, gamma, p_sat, phi, p)
    phi[0] = redlich_kwong_phi_mayus_metanol(y, p, T)
    phi[1] = redlich_kwong_phi_mayus_benceno(y, p, T)
    p = gamma_phi_p(x, gamma, p_sat, phi)

  return p, y


def calcular_wilson_A12_A21_phi_gamma(T):
  x1 = global_x1_experimentales[1:-1]
  p = global_p_experimentales[1:-1]

  def funcion_objetivo(A):
    A12, A21 = A

    s = 0

    for x1_exp, p_exp in zip(x1, p):
      x2_exp = 1 - x1_exp
      x = [x1_exp, x2_exp]

      gamma1 = wilson_gamma_1(x1_exp, A12, A21)
      gamma2 = wilson_gamma_2(x2_exp, A12, A21)

      gamma = [gamma1, gamma2]
      p_calculada, y = bublp_phi_gamma(x, gamma, T)

      diff = p_exp - p_calculada

      s += diff * diff

    return s

  def restringir_A12(A):
    return A[0]

  def restringir_A21(A):
    return A[1]

  restricciones = ({"type": "ineq", "fun": restringir_A12},
                   {"type": "ineq", "fun": restringir_A21})

  return minimize(funcion_objetivo, [0.5, 1], constraints=restricciones).x


def calcular_wilson_A12_A21_raoult_mod(T):
  x1 = global_x1_experimentales[1:-1]
  p = global_p_experimentales[1:-1]

  def funcion_objetivo(A):
    A12, A21 = A

    s = 0

    for x1_exp, p_exp in zip(x1, p):
      x2_exp = 1 - x1_exp
      x = [x1_exp, x2_exp]

      gamma1 = wilson_gamma_1(x1_exp, A12, A21)
      gamma2 = wilson_gamma_2(x2_exp, A12, A21)

      gamma = [gamma1, gamma2]
      p_calculada, y = bublp_raoult_mod(x, gamma, T)

      diff = p_exp - p_calculada

      s += diff * diff

    return s

  def restringir_A12(A):
    return A[0]

  def restringir_A21(A):
    return A[1]

  restricciones = ({"type": "ineq", "fun": restringir_A12},
                   {"type": "ineq", "fun": restringir_A21})

  return minimize(funcion_objetivo, [0.5, 1], constraints=restricciones).x


def mostrar_datos(y_calc, p_calc, x_graf, y_graf, p_graf, titulo):
  rmsep_calculado = rmsep(p_calc, global_p_experimentales)

  print(f"{titulo}: rmsep = {rmsep_calculado}")

  plt.title(titulo)
  plt.plot(x_graf, p_graf, label="punto de burbuja calculado")
  plt.plot(y_graf, p_graf, label="punto de rocio calculado")

  plt.plot(global_x1_experimentales, 
           global_p_experimentales, 
           "s", 
           label="punto de burbuja experimental")

  plt.plot(global_y1_experimentales, 
           global_p_experimentales, 
           "v", 
           label="punto de rocio experimental")

  plt.xlabel("x1-y1")
  plt.ylabel("P (kPa)")
  plt.grid()
  plt.legend()
  plt.show()


###############################################################################
# Inicio de la ejecución del codigo
###############################################################################


# Los puntos en x utilizados para graficar
x_graf = np.linspace(0, 1, 128)


# Raoult
y_calc = []
p_calc = []

for x1_actual in global_x1_experimentales:
  x = [x1_actual, 1 - x1_actual]

  P, y = bublp_raoult(x, global_t_kelvin)

  p_calc.append(P)
  y_calc.append(y[0])


y_graf = []
p_graf = []

for x1_actual in x_graf:
  x = [x1_actual, 1 - x1_actual]

  P, y = bublp_raoult(x, global_t_kelvin)

  p_graf.append(P)
  y_graf.append(y[0])

mostrar_datos(y_calc,
              p_calc,
              x_graf,
              y_graf,
              p_graf,
              "raoult")

# Raoult Modificada
A12, A21 = calcular_wilson_A12_A21_raoult_mod(global_t_kelvin)

y_calc = []
p_calc = []

for x1_actual in global_x1_experimentales:
  x = [x1_actual, 1 - x1_actual]

  gamma1 = wilson_gamma_1(x[0], A12, A21)
  gamma2 = wilson_gamma_2(x[1], A12, A21)
  gamma = [gamma1, gamma2]

  P, y = bublp_raoult_mod(x, gamma, global_t_kelvin)

  p_calc.append(P)
  y_calc.append(y[0])


y_graf = []
p_graf = []

for x1_actual in x_graf:
  x = [x1_actual, 1 - x1_actual]

  gamma1 = wilson_gamma_1(x[0], A12, A21)
  gamma2 = wilson_gamma_2(x[1], A12, A21)
  gamma = [gamma1, gamma2]

  P, y = bublp_raoult_mod(x, gamma, global_t_kelvin)

  p_graf.append(P)
  y_graf.append(y[0])

mostrar_datos(y_calc,
              p_calc,
              x_graf,
              y_graf,
              p_graf,
              "raoult mod")


# Gamma phi
A12, A21 = calcular_wilson_A12_A21_phi_gamma(global_t_kelvin)


y_calc = []
p_calc = []

for x1_actual in global_x1_experimentales:
  x = [x1_actual, 1 - x1_actual]

  gamma1 = wilson_gamma_1(x[0], A12, A21)
  gamma2 = wilson_gamma_2(x[1], A12, A21)
  gamma = [gamma1, gamma2]

  P, y = bublp_phi_gamma(x, gamma, global_t_kelvin)

  p_calc.append(P)
  y_calc.append(y[0])

y_graf = []
p_graf = []

for x1_actual in x_graf:
  x = [x1_actual, 1 - x1_actual]

  gamma1 = wilson_gamma_1(x[0], A12, A21)
  gamma2 = wilson_gamma_2(x[1], A12, A21)
  gamma = [gamma1, gamma2]

  P, y = bublp_phi_gamma(x, gamma, global_t_kelvin)

  p_graf.append(P)
  y_graf.append(y[0])

mostrar_datos(y_calc,
              p_calc,
              x_graf,
              y_graf,
              p_graf,
              "gamma-phi")


# %%
