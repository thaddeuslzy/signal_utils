import librosa as lr
import librosa.output as lo
import librosa.feature as lf
import librosa.feature.inverse as lfi
import numpy as np
import numpy.random as nr
import scipy as sp
import random

'''
################################ PRE PROCESSING ################################
'''
# def open_wav(filename, sample_rate, normal_const = 2 ** 15, DEBUG=0):
#     audio = o_audio * 1.0
#     if DEBUG > 0:
#         print("Reading from",filename)
#     # Parsing the wav file
#     f_sr,o_audio = sp.io.wavfile.read(filename)
#     print("Original Audio shape",audio.shape)

#     DOWNSAMPLE = False
#     if not f_sr == sample_rate:
#         DOWNSAMPLE = True
#         print("\nOriginal sampling aka frame rate:",f_sr)
#         # Resample to 8k Hz (Loses information)
#         # Try different resampling configs (kaiser best)
#         audio = lr.resample(audio,f_sr,sample_rate)
#         if DEBUG > 0:
#             print("Audio shape",audio.shape)

#     normalize = lambda x: x/(normal_const)
#     audio = normalize(audio)

#     original_len = audio.shape[0]

#     return audio

def lr_open_wav(filename, sample_rate, DEBUG=0):
    DOWNSAMPLE = False

    # Parsing the wav file
    signal, s_freq = lr.load(filename, sample_rate)
    audio = signal * 1.0

    # If need to resample to 8kHz
    if not s_freq == sample_rate:
        DOWNSAMPLE = True
        # Try different resampling configs (kaiser best)
        audio = lr.resample(audio, s_freq, sample_rate)

    if DEBUG > 0:
        print("Reading from",filename)
        print("Original Audio shape",audio.shape)
        print("\nOriginal sampling aka frame rate:",f_sr)
        print("Audio shape",audio.shape)

    original_len = audio.shape[0]
    return audio

# def get_power(waveform, win_len, hop_len):
#     spectro = lr.stft(waveform,
#                       n_fft=int(win_len),
#                       hop_length=hop_len,
#                       win_length=int(win_len))
#     # Assuming shape (freq, windows)
#     mag = np.abs(spectro)
#     p_frame_avg = np.average(mag**2,0) # Average power of each frame
#     p_wav = np.average(p_frame_avg) # Average power of the entire audio wave
#     return p

# In: Waveform
# Out: Spectrogram
def stft(audio):
    WINDOW_LENGTH = 256
    return lr.stft(audio,
                   n_fft=WINDOW_LENGTH,
                   hop_length=WINDOW_LENGTH//2,
                   win_length=int(WINDOW_LENGTH))
                   
# In: Waveform
# Out: MelSpectrogram
def melspectro(audio, sr=8000, win_len=256, n_mels=29):
  hop_len = win_len//2
  return lf.melspectrogram(audio, sr=sr, n_fft=win_len, hop_length=hop_len, win_length=win_len, n_mels=n_mels)
  
def melspectro_batch(wav_dict):
  spectros = {}
  keys = wav_dict.keys()
  for key in keys:
    spectros[key] = melspectro(wav_dict[key])
  return spectros
  
def log_melspectro_batch(wav_dict):
  spectros = {}
  keys = wav_dict.keys()
  for key in keys:
    spectros[key] = np.log(melspectro(wav_dict[key]))
  return spectros
  
def log_batch(spectros):
  log_spectros = {}
  keys = spectros.keys()
  for key in keys:
    log_spectros[key] = np.log(spectros[key])
  return log_spectros

def invlog(log_spectro, base=np.e):
  inv_spectro = base**(log_spectro-1e-32)
  return inv_spectro
  
# In: Dictionary of wav
# Out: Dictionary of corresponding spectrograms
def stft_batch(wav_dict):
    spectros = {}
    keys = wav_dict.keys()
    for key in keys:
        spectros[key] = stft(wav_dict[key])
    return spectros

# In: 2D Spectrogram
# Out: Power of Spectrogram
def get_power_spec(spectro):
    if not len(spectro.shape) == 2:
        raise Exception("Expected 2D spectrogram, got shape " + str(spectro.shape))
        return -1
    mag = np.abs(spectro)         # Magnitude of spectro
    frame_avg = np.average(mag**2,0)  # Avg power of spectro row-wise
    power = np.average(frame_avg)     # Avg power of entire spectro
    return power

# In: Wav
# Out: Power of wav
def get_power_wav(audio):
    spectro = lr.stft(audio,n_fft=256,hop_length=128)
    return get_power_spec(spectro)

# In: Dictionary of wavs
# Out: Average power of all wavs
def get_avg_power(waves):
    power = []
    for wave in waves.values():
      power.append(get_power_wav(wave))
    return np.average(power)

# In: Wave file, avg pwr to normalise to
# Out: Normalised Spectrogram
def normalise_power(wave, avg_pw):
    normalised = ((avg_pw/get_power_wav(wave))**0.5)*wave
    return normalised

# Input: Dictionary of wavs
# Output: Power-normalised dic of waves
def normalise_power_batch(waves):
    avg_pw = get_avg_power(waves)
    for key in waves.keys():
        waves[key] = normalise_power(waves[key],avg_pw)
    return waves
    
# Input: Mel Spec dict
# Output: Delta, DeltaDelta dicts
def get_deltas(melSpecs):
  keys = melSpecs.keys()
  deltas = {}
  deltadeltas = {}
  
  for key in keys:
    deltas[key] = lf.delta(melSpecs[key], order = 1)
    deltadeltas[key] = lf.delta(melSpecs[key], order = 2)
  return deltas, deltadeltas

def concat_vertical(arr1, arr2, arr3):
  concat = np.concatenate((arr1,arr2,arr3), axis = 0)
  return concat
  
def concat_vertical_batch(dic1, dic2, dic3):
  keys = dic1.keys()
  concats = {}
  for key in keys: 
    concats[key] = concat_vertical(dic1[key], dic2[key], dic3[key])
  return concats
    
# Generates noise with the same power as the signal
# In: tuple of audio shape, signal power, window length, hop length
# Out: noise with same power as audio, noise power
def generate_noise(signal, wl, hl):
    s_signal = stft(signal)
    noise = nr.random(signal.shape)
    noise = noise - np.average(noise) # Normalize
    s_noise = stft(noise)

    p_signal = get_power_spec(s_signal)
    p_noise  = get_power_spec(s_noise)

    # Scale waveform and power
    scaled_noise = noise*((p_signal/p_noise)**0.5)
    p_Snoise = p_noise*(p_signal/p_noise)
    return scaled_noise, p_Snoise

# Adds noise to an audio wave.
# In: 1D Audio wave, signal to noise ratio in dB
# Out: 1D Audio wave with noise, same power as original
def add_noise(audio,snr, wl=256, hl=128, DEBUG=0):
    ratio = 10 ** (snr/10)
    spectro = stft(audio)
    p_signal = get_power_spec(spectro)
    scaled_noise, p_noise = generate_noise(audio, wl, hl)

    # Scale signal and noise to match SNR.
    # Also downscales the result to match original Power
    alpha = (ratio/(ratio+1))**0.5
    beta = 1/(ratio+1)**0.5
    result = (audio*alpha + scaled_noise*beta)

    if DEBUG > 0:
      print("Power signal vs noise", p_signal, p_noise)
      print("\nNoise Scaling factor",ratio)
      print("Scales: a", alpha, "b", beta)

    return result

# Function to crop the data in a variable shape array to the shortest sample length, start decided randomly
def cutoff(data):
    std_len = min(i.shape for i in data)[0]
    new_data = []
    for sample in data:
        start = random.randint(0,int(sample.shape[0] - std_len))
        cut_sample = sample[start:start+std_len]
        new_data.append(cut_sample)
    return np.array(new_data)

# Combines several dictionaries into one.
def concat_dicts(snrs, *dicts):
  d_list = list(dicts)
  keys = list(dicts[0].keys())
  out = {}

  # Iterate through inputs
  for key in keys:
    # Iterate through dictionaries per input

    for idx in range(len(dicts)):
      new_key = key + '-snr' + str(snrs[idx])
      out[new_key] = dicts[idx][key]
  print("Total size: ", len(keys) * len(dicts))
  return out

# Crop the data in a variable shape dictionary to the shortest sample length, start decided randomly
# In: Dictionary of spectrograms
# Out: Numpy array of cropped spectrograms
def cutoff_2d_batch(data):
    keys = list(data.keys())
    std_len = min(x_trn[key].shape[1] for key in keys)
    new_data = []
    for key in x_trn:
        start = random.randint(0,int(x_trn[key].shape[1] - std_len))
        sample = x_trn[key]
        cut_sample = sample[:,start:start+std_len]
        new_data.append(cut_sample)
    return np.array(new_data)

# # For each frame, grab the context window
# # In: Dict of normal spectrograms (129, X)
# # Out: Dict of 3D TRANSPOSED spectrograms (X, 2C + 1, 129) where C = context
def batch_get_context_spectro(spectros, context, hop):
  key_list = list(spectros.keys())
  out = {}
  for k in key_list:
    curr_spec = spectros[k]
    t_spec = np.swapaxes(curr_spec,0,1) #Transpose each spectrogram
    cont_lst = get_context_spectro(t_spec, context, hop, RETURN_AS_LIST = True)

    ext = 1
    for c in cont_lst:
      new_k = k +'-'+str(ext)
      out[new_k] = c
      ext += 1

  return out

#In: Single Transposed Spectro as np array
#Out: Context window of specified frame
def get_context_frame(t_spectro, frame_num, context=5):
    frames = t_spectro.shape[0]        # Num frames
    i = frame_num
    # start padding required
    if i < context:
          num_pad = int(context - i)
          pad = np.tile(t_spectro[0], (num_pad,1))        # Generate padding
          back = np.append(pad,t_spectro[:i+1], axis = 0) # Back context + middle
    else:
        back = t_spectro[i-context:i+1]                 # Back context + middle
    # end padding is required
    if i + context > frames-1:
        num_pad = i + context - frames + 1
        pad = np.tile(t_spectro[frames-1], (num_pad,1)) # Generate padding
        front = np.append(t_spectro[i+1:],pad, axis = 0) # Front context
    else:
        front = t_spectro[i+1:i+1+context] # Front context
    context_win = np.append(back,front,axis=0)
    return context_win

#In: Single Transposed Spectrogram as np array
#Out: np array of context windows w +- 5 frames with hop length 'hop'
def get_context_spectro(t_spectro, context=5, hop=1, RETURN_AS_LIST = True):
  freqs = int(t_spectro.shape[1])    # Frequencies
  frames = t_spectro.shape[0]        # Num frames
  if RETURN_AS_LIST == True:
    chunks = []
    i = 0
    while i < frames: # while index within spectro
      # start padding is required
      if i < context:
          num_pad = int(context - i)
          pad = np.tile(t_spectro[0], (num_pad,1))        # Generate padding
          back = np.append(pad,t_spectro[:i+1], axis = 0) # Back padding + middle
      else:
          back = t_spectro[i-context:i+1]                 # Back padding + middle
      # end padding is required
      if i + context > frames-1:
          num_pad = i + context - frames + 1
          pad = np.tile(t_spectro[frames-1], (num_pad,1)) # Generate padding
          front = np.append(t_spectro[i+1:],pad, axis = 0) # Front padding
      else:
          front = t_spectro[i+1:i+1+context] # Front padding
      chunk = np.append(back,front,axis=0)
      #print("single",chunk.shape,"collection",chunks.shape)
      chunks.append(chunk)
      i+=hop
    return chunks
  elif RETURN_AS_LIST == False:
    chunks = np.empty((0,2*context+1,freqs)) # Empty 3D numpy array to store context windows
    i = 0
    while i < frames: # while index within spectro
      # start padding is required
      if i < context:
          num_pad = int(context - i)
          pad = np.tile(t_spectro[0], (num_pad,1))        # Generate padding
          back = np.append(pad,t_spectro[:i+1], axis = 0) # Back padding + middle
      else:
          back = t_spectro[i-context:i+1]                 # Back padding + middle

      # end padding is required
      if i + context > frames-1:
          num_pad = i + context - frames + 1
          pad = np.tile(t_spectro[frames-1], (num_pad,1)) # Generate padding
          front = np.append(t_spectro[i+1:],pad, axis = 0) # Front padding
      else:
          front = t_spectro[i+1:i+1+context] # Front padding
      chunk = np.expand_dims(np.append(back,front,axis=0),axis=0)
      #print("single",chunk.shape,"collection",chunks.shape)
      chunks = np.append(chunks,chunk,axis=0)
      i+=hop
    return chunks

# In: Numpy array of spectrograms
# Out: Numpy array of transposed spectrograms
def transpose_matrix(matrix):
    result = np.swapaxes(matrix,1,2)
    print("Finished transposing")
    return result

# In: Dictionary of spectrograms
# Out: Dictionary of transposed spectrograms
def transpose_batch(spectros_dict):
    transposed = {}
    keys = spectros_dict.keys()
    for key in keys:
        transposed[key] = spectros_dict[key].transpose()
    return transposed

def stft_along_axis(data, window_len = 256, hop_len = 128):
    stft = lambda x: lr.stft(x, window_len, hop_len, window_len)
    labels = np.apply_along_axis(stft,1,data)
    return labels

#In: Dictionary of spectrograms
#Out: Dictionary of normalised spectrograms
# OLD METHOD
# def normalise_spectros(inputs, f_bins, DEBUG = False):
#     # Concatenate to 1 array
#     inputs_concat = normalise_concat(inputs,f_bins, DEBUG)
#     averages = []
#     stds = []
#     rows = inputs_concat.shape[0]
#
#     averages = np.mean(inputs_concat, axis=1)      # Average across all bins
#     stds = np.std(inputs_concat, axis=1)   # Variance of all bins
#
#     # Normalize
#     for key in inputs.keys():
#       for row in range(rows):
#         inputs[key][row] = (inputs[key][row]-averages[row])/stds[row]
#
#     return inputs

# Function to get average and variance of spectrograms
# In: Spectrogram Dictionary, n frequency bands
# Out: average, variance
def get_avg_var(spec_dict, f_bands=129):
  summ = np.zeros((f_bands))
  frames = 0
  keys = list(spec_dict.keys())
  # Get average
  for k in keys:
    spec = spec_dict[k]
    summ = summ + np.sum(spec,axis=1)
    frames += spec.shape[1]
  averages = summ / frames
  flat_avg = np.reshape(averages,(f_bands,1))
  diffs = np.zeros(f_bands)
  # Getting Variance
  for k in keys:
    diff = np.square(spec_dict[k] - flat_avg)
    diff = np.sum(diff,axis=1)
    diffs += diff
  variance = diffs/frames
  return averages, variance

# Normalise single spectrogram
# In: spectrogram, average, variance
# Out: Normalised spectrogram.
def normalise_spectro(spectro, avg, var):
  s_avg = np.reshape(avg,(spectro.shape[0],1))
  s_var = np.reshape(var,(spectro.shape[0],1))
  out = (spectro - s_avg) / s_var
  return out

#In: Dictionary of spectrograms
#Out: Dictionary of normalised spectrograms and array of average and variances
def batch_normalise_spectros(spec_dict, f_bins, DEBUG = False, inPlace = False):
    out = {}
    avg, var = get_avg_var(spec_dict, f_bins)
    for k in list(spec_dict.keys()):
        if inPlace:
            spec_dict[k] = normalise_spectro(spec_dict[k], avg, var)
        else:
            out[k] = normalise_spectro(spec_dict[k], avg, var)

    return spec_dict if inPlace else out, avg, var

# #In: Dictionary of spectrograms
# #Out: Numpy arr of concatenated spectrograms
# def normalise_concat(inputs, DEBUG = False):
#   output = np.empty((129,0))
#   count = 0
#   for spectro in inputs.values():
#     output = np.append(output, spectro, axis = 1)
#     count += 1
#     if DEBUG == True:
#         print(str(count) + ' array concatenated')
#   return output

#In: Dictionary of spectrograms
#Out: Numpy arr of concatenated spectrograms
def normalise_concat(spectros, f_bins, DEBUG = False):
  length = 0
  # Obtain max length of concat
  for key in spectros.keys():
    length += spectros[key].shape[1]
    if DEBUG == True:
      print(str(length)+ 'current length')

  count = 0
  arrayth = 0
  output = np.empty((f_bins,length))

  # Concat arrays
  for spectro in spectros.values():
    output[:,count:(count+spectro.shape[1])] = spectro
    count += spectro.shape[1]
    arrayth += 1
    if DEBUG: print(str(arrayth) + ' array concatenated')
  if DEBUG: print(output.shape)
  return output

#In: Dictionary of spectrograms
#Out: Dictionry of lg_pwr spectrograms
def log_pwr_batch(data):
    keys = data.keys()
    output = {}
    for key in keys:
        output[key] = np.log(np.abs(data[key]+1e-32))
    return output

#In: Dictionary of melspec
#Out: Dictionary of lg_mel spectrograms
def log_mel(data):
  keys = data.keys()
  output = {}
  for keys in keys:
    output[key] = np.log(data[key]+1e-32)
  return output
  

'''
################################ TRAINING ################################
'''

def get_label_from_x(x_key):
  split = x_key.split('-')
  old_key = split[0]
  return old_key

# split is an integer from 1 to 9
# Out: train set, val set
def split_train_test_x(x_dataset, split):
  key_list = list(x_dataset.keys())
  key_shuffle = key_list.copy()
  random.shuffle(key_shuffle)
  random.shuffle(key_shuffle)

  x_trn = {}
  x_val = {}

  # Train/Val Split
  multiple = len(key_list)//10
  train_split = int(split*10)
  assert isinstance(train_split,int)

  input_keys = key_shuffle[:multiple*train_split]
  val_keys = key_shuffle[multiple*train_split:]

  # Generate train/validation dictionaries
  for key in input_keys:
    x_trn[key] = x_dataset[key]
  for key in val_keys:
    x_val[key] = x_dataset[key]

  return x_trn, x_val

# Cut a single frame from a random point in both input and labels.
# In: x_dictionary, y_dictionary, key, context in, context output
# Out: context x, context y
def context_from_dict(x, y, key, ctx, l_ctx):
  
  # Fetch spectrogram
  x_spectro = x[key]
  y_key = get_label_from_x(key)
  y_spectro = y[y_key]
  
  # Cut context window from spectrogram
  x_spectro = np.swapaxes(x_spectro,0,1)
  y_spectro = np.swapaxes(y_spectro,0,1)
  spec_size = x_spectro.shape[0]
  f_index = np.random.randint(0,spec_size)
  x_arr = get_context_frame(x_spectro,f_index, ctx)
  y_arr = get_context_frame(y_spectro,f_index, l_ctx)
  return x_arr, y_arr

# Generate input (x,y) for trainings
def gen_input_from_trn(x_trn, labels, bs, context=5, l_context=0):
  while True:
    x = []
    y = []
    keys_shuffle = list(x_trn.keys())
    random.shuffle(keys_shuffle)

    # Obtain batch size number of inputs
    for idx in range(bs):
      key = keys_shuffle[idx]
      x_arr, y_arr = context_from_dict(x_trn, labels, key, context, l_context)
      x.append(x_arr)
      y.append(y_arr)

    # Convert to numpy array
    x = np.array(x)
    y = np.array(y)
    yield x,y

# Generate input (x,y) for validation
def gen_input_from_val(x_val, labels,bs=256, context=5, l_context=0):
  while True:
    x = []
    y = []
    val_keys = list(x_val.keys())
    for idx in range(bs):
      key = val_keys[idx]
      # Fetch spectrogram
      x_spectro = x_val[key]
      y_key = get_label_from_x(key)
      y_spectro = labels[y_key]

      # Cut context window from spectrogram
      x_spectro = np.swapaxes(x_spectro,0,1)
      y_spectro = np.swapaxes(y_spectro,0,1)

      spec_size = x_spectro.shape[0]
      f_index = np.random.randint(0,spec_size)
      x_arr = su.get_context_frame(x_spectro,f_index, context)
      y_arr = su.get_context_frame(y_spectro,f_index, l_context)

      x.append(x_arr)
      y.append(y_arr)

    x = np.array(x)
    y = np.array(y)
    yield x,y
    
# In: Training dictionary(noisy), all labels(clean), batch size, context in, context out
# Out: (x, y) inputs , (y, x, y, x) labels
def gen_cycle_input(x_trn, labels, bs, context=5,l_context=5, FLIP=False):
  while True:
    x = []
    y = []
    keys_shuffle = list(x_trn.keys())
    random.shuffle(keys_shuffle)

    # Obtain batch size number of inputs
    for idx in range(bs):
      key = keys_shuffle[idx] # 'input_1-snr0'
      x_arr, y_arr = context_from_dict(x_trn, labels, key, context, l_context)
      x.append(x_arr)
      y.append(y_arr)

    # Convert to numpy array
    x = np.array(x)
    y = np.array(y)
    # outputs: (x1, x2, y1, y2)
    if FLIP:
      yield ((y, x), (y, x, y, x))
    else:
      yield ((x, y), (y, x, y, x))
        
'''
################################ POST PROCESSING ################################
'''

# Combine a wave amplitude and phase
def join_wave(amp,phs):
    def pol_to_cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    # combine
    a, b = pol_to_cart(amp,phs)
    return a + b*1j

def wav_to_file(out_path, wav, sample_rate = 8000):
  o_wave = wav.astype('float32')
  lo.write_wav(out_path,o_wave,sample_rate)
  print("Written to " + out_path)

# Feeding in a non-complex spectrogram will return a wave with zero phase
# In: log spectrogram
# Out: returns a waveform
def spectro_to_wav(spectrogram, sr = 8000, window_len = 256, hop_len = 128, phase=-1):
  # Reverse the absolute function to get back a cartesian form complex number
  if isinstance(phase,int):
    phase = np.zeros(spectrogram.shape)
  joined = join_wave(spectrogram,phase)
  # Get the length of the wav in ms
  wav_length = int(spectrogram.shape[1]*hop_len)
  amplitude = lr.istft(joined,hop_length=hop_len,win_length=window_len,length=wav_length,center=True)
  return amplitude

# In: Trained model, data as a tuple, number of frequency bands, number of outputs...
# Out: A list of predicted spectrograms by the given model from the given data.
# Stitched together and scaled down for overlaps
def multi_predict(model, data, f_bands, outputs=4, radius=5, hop=1, out_radius=5):
  num_windows = data[0].shape[0]
  end_shape = (num_windows*hop + 2*out_radius, f_bands)
  zero_arr = np.zeros(end_shape)
  multi_out = np.tile(zero_arr, (outputs,1,1))

  ci = out_radius # center index
  # Iterate over each context window
  for i in range(num_windows):
    inp = ()
    for d in data:
      frame = d[i]
      frame = np.reshape(frame,(1,frame.shape[0],frame.shape[1]))
      inp = inp + (frame,)
    raw_pred = model.predict(inp)
    for op in range(outputs):
      # Add prediction to spectro
      out_arr = multi_out[op]
      pred = raw_pred[op]
      out_arr[ci-out_radius: ci + out_radius + 1] = out_arr[ci-out_radius: ci + out_radius + 1] + pred
    ci += hop

  final_out = []
  for i in range(outputs):
    pred_arr = multi_out[i][out_radius:-out_radius]
    #Fringe scaling
    for i in range(out_radius):
      pred_arr[i] = pred_arr[i]/(out_radius+i+1)
      pred_arr[-i-1] = pred_arr[-i-1]/(out_radius+i+1)
    #Main body scaling
    pred_arr[out_radius:-out_radius] = pred_arr[out_radius:-out_radius]/(2*out_radius + 1)
    pred_arr = np.swapaxes(pred_arr,0,1)
    final_out.append(pred_arr)

  print('Final out shape',final_out[0].shape)
  return final_out

# Unnormalise spectro. It is still log
def unnormalise_spectro(spectro, avg, var):
  s_avg = np.reshape(avg,(spectro.shape[0],1))
  s_var = np.reshape(var,(spectro.shape[0],1))
  out = (spectro*s_var) + s_avg
  return out

# Unlog the spectrogram
def restore_spectro(spectro, avg, var, base = np.e):
  log_spec = unnormalise_spectro(spectro, avg, var)
  original = base**(log_spec-1e-32)
  return original
  
def restore_spectro_mel(spectro, avg, var, base = np.e):
  log_spec = unnormalise_spectro(spectro, avg, var)
  original = base**(log_spec)
  return original
