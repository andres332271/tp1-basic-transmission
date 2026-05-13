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

A continuación veremos el plotteo obtenido:

![rtaImpulsos](/images/rc_01_impulse_response.png)

Podemos observar varias señales del tipo `sinc` superpuestas entre sí, cada una con un valor de roll-off ($\beta$) diferente. Lo que tienen **en común** estas señales es que todas, en los instantes de tiempo normalizados, la amplitud del filtro es cero para cualquier valor de $\beta$. Estos puntos donde la señal cruza con el eje son los momentos en los cuales el receptor lee los símbolos adyacentes, de esta forma garantizamos que no se interfiera en la lectura de otros símbolos. Y en lo que se **diferencian** estas señales es en la amplitud a lo largo del tiempo, es decir, a medida que nos alejamos del origen aquellas señales con $\beta$ mayor disminuirán más sus picos que aquellas con uno menor.

Para concluir el estudio de este filtro se hizo uso de la función `numpy.fft.fft()` y se graficó la magnitud de la respuesta en frecuencia de todas las respuestas al impulso calculadas en el punto anterior. El plotteo obtenido es el siguiente:

![rtaFrecuencia](/images/rc_02_freq_response.png)

Podemos observar que lo que tienen **en común** todas estas respuestas es que todas cortan exactamente en el mismo punto, en el punto de la *frecuencia de Nyquist*, que a su vez equivale a la mitad de la amplitud ($0.5$). Por otro lado, todas **se diferencian** en cuanto ancho de banda total ocupan, esto se debe a los diferentes valores de $\beta$ de cada señal, a mayor $\beta$ mas ancho de banda del espectro consumimos y el filtro se ensancha.

También se realizó el plotteo para la respuesta en frencuencia en $[dB]$: 

![rtaFrecuenciadB](/images/rc_03_freq_response_db.png)

