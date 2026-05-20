# Trabajo Práctico N°1 - Comunicaciones Ópticas

## Ejercicio 1

Para este ejercicio se diseñó mediante código en Python un filtro del tipo Coseno Realzado (RC), para el mismo se creó una función llamada `raised_cosine()` que recibe como parámetros: 

   - El *symbol rate* (equivalente a 1/T),
   - La frecuencia de muestreo del filtro,
   - El *rolloff* o exceso de ancho de banda β,
   - La cantidad de *taps* que se desean calcular.

Y nos devuelve la respuesta al impulso del filtro. Para realizar un análisis de la misma se generó una figura donde se superponen varias respuestas al impulso para: 

   - BR: 32 GBd
   - fs: 4\*BR
   - β variable

A continuación veremos el plot obtenido:

![rtaImpulsoRC](/images/rc_01_impulse_response.png)

En la imagen se observa como cambia la respuesta al impulso del filtro RC al aumentar el factor de rolloff desde 0.1 hasta 0.9. Para valores altos la amplitud cae más rápidamente a medida que nos alejamos de t = 0, con lo que podría decirse que hacen falta menos coeficientes de filtro para obtener una representación fiel. A su vez todas las curvas comparten dos características:
- a) En todas las curvas la respuesta es cero para instantes de tiempo t*BR = k*T (valores enteros de tiempo normalizado), excepto en t=0. T es el período de símbolo, por lo que t*BR = k*T, k!=0 representan los instantes de tiempo donde se transmiten los símbolos adyacentes en la trama de comunicación. Al valer cero nos aseguramos que los símbolos no se interfieren entre sí.
- b) El pico en t = 0 vale 0.25 para todas las curvas. En la teoría el valor pico de este tipo de filtros es 1, pero al utilizar sobremuestreo con N=4 debemos escalar la amplitud para mantener la ganancia del filtro en 1 (0 dB).

Para concluir el estudio de este filtro se hizo uso de la función `numpy.fft.fft()` y se graficó la magnitud de la respuesta en frecuencia de todas las respuestas al impulso calculadas en el punto anterior. El plot obtenido es el siguiente:

![rtaFrecuenciaRC](/images/rc_02_freq_response.png)

Podemos observar que lo que tienen **en común** todas estas respuestas es que todas cortan exactamente en el mismo punto, en el punto de la *frecuencia de Nyquist*, que a su vez equivale a la mitad de la amplitud ($0.5$). Por otro lado, todas **se diferencian** en cuánto ancho de banda total ocupan, esto se debe a los diferentes valores de $\beta$ de cada señal, a mayor $\beta$ mas ancho de banda del espectro consumimos y el filtro se ensancha.

También se realizó el plotteo para la respuesta en frencuencia en $[dB]$: 

![rtaFrecuenciaRC_dB](/images/rc_03_freq_response_db.png)


## Ejercicio 2

Para este ejercicio se utilizaron los mismos parámetros que en el anterior, pero sobre filtros RRC (Root Rised Cosine). Primero analizamos la respuesta en frecuencia del filtro:

![rtaImpulsoRRC](/images/rrc_01_impulse_response.png)

Vemos que ahora las curvas no tienen nada en común, a lo sumo la forma, pero no comparten ningún punto. Si se las compara con las respuestas de los filtros RC podría decirse que:
- Tanto en las curvas de respuesta al impulso de filtros RC como en las de filtros RRC al aumentar el factor de rolloff la amplitud disminuye más rápidamente a medida que nos alejamos de t=0.
- A diferencia de la respuesta al impulso del filtro RC, el filtro RRC no presenta respuesta cero en t*BR=kT. Esto tiene como consecuencia que este filtro introduce interferencia entre símbolos (ISI).
- Otra diferencia con respecto al filtro RC es que en la respuesta al impulso del filtro RRC al aumentar el factor de rolloff el valor del pico en t=0 aumenta. Esto probablemente se deba a que la respuesta en frecuencia del filtro RRC se obtiene al tomar la raíz cuadrada de la respuesta en frecuencia del filtro RC (de ahí su nombre). El valor h[0] representa el area bajo la curva del la respuesta en frecuencia. Como dicha curva se mantiene por debajo de 1 (ganancia unitaria) al aplicar la raíz cuadrada el valor resultante es ligeramente más alto. Al aumentar el factor de rolloff la respuesta en frecuencia se ensancha, al igual que en filtros RC, pero al tener un punto de "pivot" (en f=BR/2) más alto (0.707 para RRC comparado con 0.5 para RC) la curva encierra más área y eso explica el aumento de h[0].

A continuación se presenta la respuesta en frecuencia del filtro RRC

![rtaFrecuenciaRRC](/images/rrc_02_freq_response.png)

Como se observa, el efecto de tomar la raíz cuadrada sobre la respuesta en frecuencia del filtro RC es aumentar el nivel de la curva, ya que toda la curva pasa por debajo de 1. Esto se ve claramente en el punto que tienen en común todas las curvas para distintos factores de rolloff, en f=BR/2, que en un filtro RC vale 0.5 y en un filtro RRC vale 0.707.

Al igual que ocurre con el filtro RC, un filtro RRC con mayor rolloff ocupará más ancho de banda. Es más, dados dos filtros, uno RC y otro RRC con el mismo factor de rolloff, el filtro RRC ocupa mayor ancho de banda.

Adicionalmente, se hizo un plot de la respuesta en frecuencia con la escala de amplitud en dB.

![rtaFrecuenciaRRC_dB](/images/rrc_03_freq_response_db.png)

Para verificar que al convolucionar dos filtros RRC obtenemos un filtro RC hicimos un plot del error, con curvas para distintos valores de rolloff. El error es simplemente la diferencia entre la respuesta al impulso de un filtro RC y la convolución de dos RRC (error absoluto). Este error se puede caracterizar tanto en el tiempo como en frecuencia.

A continuación se presentan los plots del error en tiempo y frecuencia.


![rtaError2RRCvsRC](/images/rrc_04_verification_time.png)
![rtaError2RRCvsRC_frec](/images/rrc_05_verification_freq.png)

A partir de estos resultados podemos hacer algunos comentarios:
- El error máximo es bajo (menor a 1e-3) con lo que a grandes rasgos podemos decir que convolucionar dos filtros RRC nos da como resultado un filtro RC.
- Al aumentar el rolloff el error máximo disminuye, por lo que con un rolloff alto la convolución RRC*RRC será más fiel al RC original, aunque esto es a costa de consumir un mayor ancho de banda.
- Los picos de error se dan en t=0 y t=+-t_max/2 y alrededor de la frecuencia de Nyquist (f=BR/2) en frecuencia.

Para acentuar aún más estos patrones se aumentaron los taps del filtro de 101 a 501. Estos son los resultados del error:

![rtaError2RRCvsRC_2](/images/rrc_04_verification_time_1.png)
![rtaError2RRCvsRC_frec_2](/images/rrc_05_verification_freq_1.png)

Una observación adicional. Al aumentar los taps, el error del producto RRC*RRC también se reduce, cosa que era de esperar.
