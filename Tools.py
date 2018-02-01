# Se importan las librerias utilizadas.

import numpy as np
import librosa
from librosa.core.time_frequency import frames_to_samples, time_to_samples
from librosa.util.exceptions import ParameterError

# Se define el objeto "stimulus" utilizado para generar los estimulos de los experimentos, 
# con sus respectivas entradas.

def stimulus(isi,rep,pause,fs = 44100,click_freq=[1000.0]):
    
    times = np.arange(rep)*isi/1000.0
    
    s2 = pause + times[-1]*2
    
    out = clicks(times,sr = fs,click_freq=click_freq)
    out = np.r_[out,np.zeros(int(s2*fs))]
    out = np.c_[out,out]
    return out

# Se define el objeto "isocrono", que genera estimulos cuyos bips tienen
# todos la misma frecuencia. 

def isocrono(isi,rep,pause,fs = 44100):
    
    times = np.arange(rep)*isi/1000.0
    
    out = librosa.clicks(times,sr = fs)
    out = np.r_[out,np.zeros(int(pause*fs))]
    out = np.c_[out,out]
    return out

# Se define el objeto "clicks" utilizado para generar los bips de distintas frecuencias
# del experimento 2.

def clicks(times=None, frames=None, sr=22050, hop_length=512,
           click_freq=[1000.0], click_duration=0.1, length=None):

    from librosa.core.time_frequency import frames_to_samples, time_to_samples
    from librosa.util.exceptions import ParameterError

    
    if times is None:
        if frames is None:
            raise ParameterError('either "times" or "frames" must be provided')

        positions = frames_to_samples(frames, hop_length=hop_length)
    else:
        # Convert times to positions
        positions = time_to_samples(times, sr=sr)


    if click_duration <= 0:
        raise ParameterError('click_duration must be strictly positive')

    # Set default length
    if length is None:
        length = positions.max()+22050
    else:
        if length < 1:
            raise ParameterError('length must be a positive integer')

        # Filter out any positions past the length boundary
        positions = positions[positions < length]

    # Pre-allocate click signal
    click_signal = np.zeros(length, dtype=np.float32)

    # Place clicks
    c = -1
    for start in positions:

        c+=1
        if c==len(click_freq):
            c=0

        angular_freq = 2 * np.pi * click_freq[c] / float(sr)

        click = np.logspace(0, -10,
                            num=int(np.round(sr * click_duration)),
                            base=2.0)

        click *= np.sin(angular_freq * np.arange(len(click)))


        # Compute the end-point of this click
        end = start + click.shape[0]

        if end >= length:
            click_signal[start:] += click[:length - start]
        else:
            # Normally, just add a click here
            click_signal[start:end] += click

    return click_signal

# Se define el objeto "peaks" utilizado para detectar los taps de los participantes.

def peaks(y, thres=0.3, min_dist=1):
 
    dy = np.diff(y)
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks

