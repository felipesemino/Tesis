# Se importan las librerias utilizadas y los objetos creados en el archivo "tools.py".

import sounddevice as sd
from itertools import product
import librosa
import pandas as pd
from random import shuffle
import numpy as np
import time
from tools import stimulus, peaks
from matplotlib import pyplot as plt

# Checkea el audio.

print('Using device {}'.format(sd.default.device))

PLOT_RESULTS = True

# Seleccion manual de Frecuencia de Sampleo y Canales de Audio. 

fs = 44100
response_channel = 0
loopback_channel = 1
tap_threshold = 0.1

# Define objetos

dur_total = 0
pause = 3

# Detecta la fecha y la hora en la que se realiza el experimento.

fecha = time.strftime("%d-%m-%Y_%H-%M")

# Pide al participante ingresar una sigla de 3 letras relacionada a su nombre.

print ("Escribi una sigla de 3 letras de tu nombre")
nombre = input()

# Checkea que sean 3 letras.

while len(nombre) != 3 :
    print ("No son 3 letras")
    print ("Escribi una sigla de 3 letras de tu nombre")
    nombre = input()
nombre = str(nombre)
print ("Hola "+nombre)


# Se establecen manualmente las condiciones de los estimulos.

n_trial_per_condition = 1
n_trial_per_condition = list(range(n_trial_per_condition))
n_sub = [3,4,6,8]
total_interval = 2400

conditions = product(n_sub,n_trial_per_condition)
conditions = list(conditions)

dur_total = len(conditions)*(total_interval/1000.0*2+pause)

# Imprime la duracion total estimada del experimento.

print('Duracion {} minutos'.format(dur_total/60.0))

shuffle(conditions)

respuestas_full = []
respuestas = []

# Reproduce un sonido para activar la libreria de audio.

sd.play(np.zeros(1000),samplerate = fs)
plt.pause(2)

# Pide presionar Enter para comenzar el experimento.

print ("Presiona Enter para comenzar") 
input()
print ("Empieza")


# Reproduce los estimulos y graba las respuestas a travez de una estructura "for loop".

for j_trial, (subs, _) in enumerate(conditions):

    isi = total_interval/subs
    stim = stimulus(isi,subs+1,2)

    print('Secuencia:',j_trial+1,'de', len(conditions) )

    not_finish = True

    while not_finish:

        plt.pause(1) 

        record = sd.playrec(stim, fs, channels=2, blocking=True) 
        plt.pause(0.2)

# Chequea cantidad de taps de cada respuesta y repite el estimulo en el caso de que no sea la 
# cantidad esperada.

        resp = record[:,response_channel].flatten()
        loopback = record[:,loopback_channel].flatten()

        min_dist = int(isi/1000*fs/2.0)
        taps_idx = peaks(resp,thres=tap_threshold,min_dist=min_dist)
        n_taps = len(taps_idx)        

        if n_taps==subs+1:
            not_finish=False
        else:
            pass

    respuestas_full.append({'respuesta': resp,
                            'isi'      : isi,
                            'estimulo' : stim,
                            'loopback' : loopback,
                            'subs'     : subs,
                            'n_taps'   : n_taps,
                            't_taps'   : taps_idx,
                            'fs'       : fs, 
                            'subject':nombre } )

    respuestas.append({'isi'   : isi,
                       'subs'  : subs,
                       'n_taps': n_taps,
                       't_taps': taps_idx,
                       'fs'    : fs, 
                       'subject':nombre } )
    
# Genera la pausa a la mitad del experimento para que el participante descanse.

    if j_trial==int(len(conditions)/2-1):
        print ("Llegaste a la mitad del experimento, descansa y presiona enter para terminarlo.")
        input ()

print("Terminaste!")

# Graba los resultados en un archivo de extension ".pickle".

df = pd.DataFrame(respuestas)
df.to_pickle('respuestas/subdivision_'+nombre+'_'+fecha+'.pickle')

filename = 'respuestas_full/full_subdivision_'+nombre+'_'+fecha+'.pickle'
df_full = pd.DataFrame(respuestas_full)
df_full.to_pickle(filename)

print('Datos salvados')

# Calcula los resultados.

if PLOT_RESULTS:

    def MU_Y_SD(l):
        x = np.vstack(l.values)

        s = np.std(x,0)
        u = np.mean(x,0)

        return pd.Series({'u':u,'s':s})

    for i,r in df.iterrows():
        x = (r.t_taps[1:] - r.t_taps[0])/r.fs
        df.set_value(i,'t_taps',x )

    gpby = df.groupby(['subs']).t_taps
    df2 = gpby.apply(MU_Y_SD).unstack()

    plt.figure(figsize=(18,8))

    plt.subplot(1,2,1)
    for i,r in df2.iterrows():
        t = np.arange(1,i+1)*2.4/(i)
        y = r.u-t
        plt.plot(t,y,'.-',label=i)
        plt.show()


    plt.legend(loc=2,title='Subdivisiones');
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Exactitud  (segundos)')
    plt.grid()

    plt.subplot(1,2,2)
    for i,r in df2.iterrows():
        t = np.arange(1,i+1)*2.4/(i)
        y = r.s

        plt.plot(t,y,'.-',label=i)
        plt.show()

# Grafica los resultados.        

    plt.legend(title='Subdivisiones');
    plt.xlabel('Tiempo (segundos)');
    plt.ylabel('Dispersión (segundos)');
    plt.grid()
    plt.tight_layout();
    plt.show();