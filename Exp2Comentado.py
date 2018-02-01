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
from scipy.io import wavfile
import os
import platform

# para poder ejecutar el programa en Linux es necesario utilizar Jack:

if platform.system()=='Linux':
    for i,s in enumerate(str(sd.query_devices()).split("\n")):
        if 'system, jack' in s.lower():
            device_idx = i
            break

    if device_idx!=None:
        # Cambiar id de disposivo si es necesario.
        sd.default.device = device_idx
    else:
        print('JACK not running')
        exit()



PLOT_RESULTS = True

# Seleccion manual de Frecuencia de Sampleo y Canales de Audio. 

fs = 44100
response_channel = 0
loopback_channel = 1
tap_threshold = 0.1

# Define objetos

F0 = 500
dur_total = 0
wait_time = 4
total_interval = 2400
pause = 200

# Se establece manualmente la cantidad de trials por condicion de la sesion de práctica.

n_trial_per_condition_train = 1
n_trial_per_condition_train = list(range(n_trial_per_condition_train))

# Se establece manualmente la cantidad de trials por condicion del experimento.

n_trial_per_condition = 24
n_trial_per_condition = list(range(n_trial_per_condition))

# Se establecen manualmente las condiciones de los estímulos.

n_sub = [4,8]
note_intervals = [0,5,10]

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

# Se especifica la dirrección y el formato con el que se guardarán los archivos 
# de audio enteros de las respuestas.

directory = 'wavs/{}_{}/'.format(nombre,fecha)

if not os.path.exists(directory):
    os.makedirs(directory)

# Reproduce un sonido para activar la libreria de audio.

sd.play(np.zeros(1000),samplerate = fs)
sd.sleep(2000)

# Sesion de práctica.

conditions = product(n_sub,note_intervals,n_trial_per_condition_train)
conditions = list(conditions)

if len(n_trial_per_condition_train)>0:

    shuffle(conditions)

# Pide presionar Enter para comenzar la sesion de práctica.

    print ("Presiona Enter para comenzar la sesion de practica") 
    input()
    print ("Empieza")

# Reproduce los estimulos y graba las respuestas a travéz de una estructura "for loop".

    for j_trial, (subs, note_interval, _) in enumerate(conditions):

        isi = total_interval/subs
        
        click_freq = [F0,F0*2**(note_interval/12.0)]
        
        stim = stimulus(isi,subs+1,2,click_freq = click_freq)

        print('Secuencia:',j_trial+1,'de', len(conditions) )

        not_finish = True

        while not_finish:

            record = sd.playrec(stim, fs, channels=2, blocking=True) #reproduce audio y graba
            
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

            sd.sleep(pause)

# Experimento.

conditions = product(n_sub,note_intervals,n_trial_per_condition)
conditions = list(conditions)

dur_total = len(conditions)*(total_interval/1000.0*2+wait_time)

shuffle(conditions)

respuestas_full = []
respuestas = []

# Pide presionar Enter para comenzar el experimento.

print ("Presiona una tecla para comenzar el experimento") 
input()
print ("Empieza")

# Reproduce los estimulos y graba las respuestas a travez de una estructura "for loop".

for j_trial, (subs, note_interval, _) in enumerate(conditions):

    isi = total_interval/subs
    
    click_freq = [F0,F0*2**(note_interval/12.0)]
    
    stim = stimulus(isi,subs+1, 2 , click_freq = click_freq)

    print('Secuencia:',j_trial+1,'de', len(conditions) )

    not_finish = True

    rep_count = 0

    while not_finish:

        record = sd.playrec(stim, fs, channels=2, blocking=True) #reproduce audio y graba
        
# Chequea cantidad de taps de cada respuesta y repite el estimulo en el caso de que no sea la 
# cantidad esperada.

        resp = record[:,response_channel].flatten()
        loopback = record[:,loopback_channel].flatten()

        min_dist = int(isi/1000*fs/2.0)
        taps_idx = peaks(resp,thres=tap_threshold,min_dist=min_dist)
        n_taps = len(taps_idx)
        
        if n_taps==subs+1:
            not_finish=False
            valid = True
        else:
            valid = False

        fname = '{}_{}.wav'.format(j_trial,rep_count)

        wavfilename = directory+fname

        wavfile.write(wavfilename,fs,record)

        respuestas.append({'wavfile'        : wavfilename,
                           'rep_count'      : rep_count,
        	               'isi'            : isi,
                           'subs'           : subs,
                           'note_interval'  : note_interval,
                           'F0'             : F0,
                           'n_taps'         : n_taps,
                           't_taps'         : taps_idx,
                           'valid'          : valid,
                           'fs'             : fs, 
                           'subject'        : nombre } )
    
        sd.sleep(pause)

        rep_count += 1

# Genera la pausa a la mitad del experimento para que el participante descanse.

    if j_trial==int(len(conditions)/2-1):
        print("Llegaste a la mitad del experimento, descansa y presiona cualquier tecla para seguir.")
        input()


print("Terminaste!")

# Graba los resultados en un archivo de extension ".pickle".

df = pd.DataFrame(respuestas)
df.to_pickle('respuestas/subdivision_notas_'+nombre+'_'+fecha+'.pickle')

print('Datos salvados')

# Calcula los resultados.

if PLOT_RESULTS:

    df = df[df['valid']]

    def MU_Y_SD(l):
        x = np.vstack(l.values)

        s = np.std(x,0)
        u = np.mean(x,0)

        return pd.Series({'u':u,'s':s})

    for i,r in df.iterrows():
        x = (r.t_taps[1:] - r.t_taps[0])/r.fs
        df.set_value(i,'t_taps',x )

    gpby = df.groupby(['subs','note_interval']).t_taps
    df2 = gpby.apply(MU_Y_SD).unstack()

    plt.figure(figsize=(18,8))
   
    for  (i,n),r in df2.iterrows():
        t = np.arange(1,i+1)*2.4/(i)
        y = r.u-t

        cix = ((i/4-1)*4+n/5)/20

        plt.subplot(1,2,1)
        plt.plot(t,y,'.-',label = '{}, {}'.format(n,i),color = plt.cm.tab20c(cix) )              

        plt.subplot(1,2,2)
        plt.plot(t,r.s,'.-',label = '{}, {}'.format(n,i),color = plt.cm.tab20c(cix) )

# Grafica los resultados.   

    plt.subplot(1,2,1)
    plt.legend().set_title('Intervalo Notas / Subdivisiones',prop={'size':18});
    plt.grid()
    plt.ylabel('Exactitud (segundos)')

    plt.subplot(1,2,2)
    plt.legend().set_title('Intervalo Notas / Subdivisiones',prop={'size':18});
    plt.ylabel('Dispersión (segundos)');
    plt.grid()

    plt.tight_layout()
    plt.show();