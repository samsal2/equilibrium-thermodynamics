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
#|\   ___ \    |\  ___ \     |\  \          |\  ___ \     |\   __  \     
#\ \  \_|\ \   \ \   __/|    \ \  \         \ \   __/|    \ \  \|\  \    
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
###############################################################################
# Declaración de constantes globales
###############################################################################

###############################################################################
# Declaración de metodos
# #############################################################################


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


def redlich_kwong_equacion_cubica_de_estado(A, B, P, Z):
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


def redlich_kwong_equacion_cubica_de_estado_derivada(A, B, P, Z):
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


def redlich_kwong_resolver_equacion_cubica_de_estado(A, B, P, Z0=1e-20, error=5e-10):
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
  error : float
    El error relativo

  Devuelve
  --------
  float
    El valor de Z encontrado
  """

  Z_pasada = Z0

  # Se va a utilizar newton-rapson
  # Z_i+1 = Z_i - f(Z_i) / f'(Z_i)
  Z_actual = Z_pasada - redlich_kwong_equacion_cubica_de_estado(A, B, P, Z_pasada) /\
      redlich_kwong_equacion_cubica_de_estado_derivada(A, B, P, Z_pasada)

  while abs((Z_actual - Z_pasada) / Z_actual * 100) > error:
    Z_pasada = Z_actual
    Z_actual = Z_pasada - redlich_kwong_equacion_cubica_de_estado(A, B, P, Z_pasada) /\
        redlich_kwong_equacion_cubica_de_estado_derivada(A, B, P, Z_pasada)

  return Z_actual


def regla_de_mezcla_A(A, x):
  """
  Regla de mezclado del parametro A descrita por la ecuación 4-42 del seader

  A = ΣAixi 

  Parametros
  ----------
  A : list[float]
    Lista con los valores de A en cada posición de i
  x : list[float]
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
  assert len(A) == len(x)

  # Valor inicial de la sumatoria
  s = 0

  # Sumar los productos
  for Ai, xi in zip(A, x):
    s += Ai * xi

  return s


def regla_de_mezcla_B(B, x):
  """
  Regla de mezclado del parametro B descrita por la ecuación 4-43 del seader

  B = ΣBixi 

  Parametros
  ----------
  B : list[float]
    Lista con los valores de B en cada posición de i
  x : list[float]
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
  assert len(B) == len(x)

  # Valor inicial de la sumatoria
  s = 0

  # Sumar los productos
  for Bi, xi in zip(B, x):
    s += Bi * xi

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


def coeficiente_de_fugacidad_para_mezcla_liquida(Zl, A, Ai, B, Bi, P):
  """
  Metodo para calcular la fugacidad descrita de la ecuación 4-73 del seader

  φiL = e^[(Zl - 1)Bi/B - ln(Zl - BP) - A^2/B(2Ai/A - Bi/B)ln(1 + BP/Zl)]

  Parametros
  ----------
  Zl : float
    Factor de compresiblidad liquida
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
  # φiL = e^[(Zl - 1)Bi/B - ln(Zl - BP) - A^2/B(2Ai/A - Bi/B)ln(1 + BP/Zl)]
  # a = (Zl - 1)Bi/B
  # b = -ln(Zl - BP)
  # c = -A^2/B
  # d = 2Ai/A - Bi/B
  # g = ln(1 + BP/Zl)
  # φiL = e^[a + b + c * d * g]

  # Precalculando valores para simplificar
  a = (Zl - 1) * Bi / B
  b = -np.log(Zl - B * P)
  c = -A * A / B
  d = 2 * Ai / A - Bi / B
  g = np.log(1 + B * P / Zl)

  return np.exp(a + b + c * d * g)


def calcular_coeficiente_de_fugacidad_para_la_substancia_i(Pc, Tc, x, P, T, i):
  """
  Metodo para calcular la fugacidad de una substancia i en una mezcla

  Parametros
  ----------
  Pc : list[float]
    Una lista con las presiónes criticas para i
  Tc : list[float]
    Una lista con las temperatura criticas para i
  x  : list[float]
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
  assert len(Tc) == len(Pc) == len(x)

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
  A = regla_de_mezcla_A(lista_Ai, x)
  B = regla_de_mezcla_B(lista_Bi, x)

  # Calcular la Z liquida
  Z = redlich_kwong_resolver_equacion_cubica_de_estado(A, B, P)

  # Initializar el valor de A y B de la substancia i
  # Las listas en python comienzar a contar desde 0
  Ai = lista_Ai[i - 1]
  Bi = lista_Bi[i - 1]

  return coeficiente_de_fugacidad_para_mezcla_liquida(Z, A, Ai, B, Bi, P)


g_pc = [42.48, 33.7]
g_tc = [368.8, 469.7]
g_x = [0.5, 0.5]
g_p = 2
g_t = 75 + 273.15


print(calcular_coeficiente_de_fugacidad_para_la_substancia_i(
    g_pc, g_tc, g_x, g_p, g_t, 1))
