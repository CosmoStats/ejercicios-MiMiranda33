La carpeta PyMultiNest_Miguel contiene los scripts para correr pyMultiNest los cuales se corren con los datos de Pantheon+SH0ES.dat y Pantheon+SH0ES_STATONLY.cov
  Los scripts FlatLambdaCDM_PyMultiNest.py, LambdaCDM_PyMultiNest.py y FlatwCDM_PyMultiNest.py infieren los parametros respectivos del modelo, siendo que en todos los modelos se infiere H_0 y \Omega_M, y los parámetros nuisance \alpha y \beta.
  Al correr los programas estos se guardan en una carpeta llamada chains automáticamente y tienen un número de iteraciones predeterminado
  Es posible correr los scripts dandole cierto número de iteraciones para inferir los parámetros con el suguiente comando:
    python FlatLambdaCDM_PyMultiNest.py --nlive 1000
  Es posible guardar los resultados generando una carpeta nueva chains-NOMBRE mediante el comando:
    python FlatwCDM_PyMultiNest.py --chains-dir chains-NOMBRE
  El script para comparar modelos mediante los criterios AIC y BIC toma los datos directamente de los archivos de las carpetas individuales chains-MODEL, y con ellos hace las compraraciones.
  Dentro de los resultados de las carpetas chains se genera un gráfico tipo corner donde se presentan los resultados.
