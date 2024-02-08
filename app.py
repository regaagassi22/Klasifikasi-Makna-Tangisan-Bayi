from flask import Flask, request, render_template
import numpy as np
from scipy.fftpack import dct
import pickle
import librosa as lbr
import uuid
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import csv

app = Flask(__name__)
model_path = 'hasil_klasifikasi_knncoba.sav'
knn_model = pickle.load(open(model_path, 'rb'))

train_data = []
num_ceps = 20

with open('data_latihcoba.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Ambil fitur MFCC (kolom 1 hingga 20)
        mfcc_features = np.array([float(row[f'mfcc{i}']) for i in range(1, 21)])
        label = row['label']  # Ambil label dari kolom 'label'

        # Tambahkan data ke train_data dalam format yang sesuai
        train_data.append({'mfcc': mfcc_features, 'label': label})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav'}

def bacadata(mp):
  signal,sr  = lbr.load(mp,duration=30,sr=22050)
  return signal,sr

def initialize(mp):
  signal,sr = lbr.load(mp,duration=30,sr=22050)
  signal = signal[0:int(30 * sr)]
  return sr,signal

def lowPassFilter(signal, pre_emphasis=0.97):
	return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

def preEmphasis(mp):
    sr, signal = initialize(mp)
    emphasizedSignal = lowPassFilter(signal)
    
    # Simpan gambar pre-emphasis menggunakan Pillow
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Original Signal')
    plt.plot(emphasizedSignal, label='Emphasized Signal')
    plt.legend()
    
    # Simpan gambar sebagai file Bytes
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    
    # Simpan gambar ke direktori "static"
    image_path = os.path.join('static', 'pre_emphasis.png')
    plt.savefig(image_path, format='png')
    return emphasizedSignal
	

def framing(mp, frame_index=100):
    windowSize = 0.025
    windowStep = 0.01
    sr, signal = initialize(mp)
    frame_length, frame_step = windowSize * sr, windowStep * sr
    signal_length = len(preEmphasis(mp))
    overlap = int(round(frame_length))
    frameSize = int(round(frame_step))
    numberOfframes = int(np.ceil(float(np.abs(signal_length - frameSize)) / overlap ))
    pad_signal_length = numberOfframes * frameSize + overlap
    if pad_signal_length >= signal_length:
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(preEmphasis(mp), z)
    else:
        pad_signal = preEmphasis(mp)

    indices = np.tile(np.arange(0, overlap), (numberOfframes, 1)) + np.tile(np.arange(0,
                numberOfframes * frameSize, frameSize), (overlap, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    # Menunjukkan hanya satu frame (misalnya, frame ke-100) dan hasil windowingnya
    frame_to_show = frames[frame_index]

    # Menampilkan hasil frame dan posisi frame
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Original Signal')
    # Menampilkan kotak berwarna sepanjang frame ke-100
    frame_start = frame_index * frameSize
    frame_end = (frame_index + 1) * frameSize
    plt.axvspan(frame_start, frame_end, facecolor='blue', alpha=0.5)
    plt.plot(signal, linewidth=1)  # Merubah warna gelombang menjadi merah
    plt.title(f'Posisi Frame ke- {frame_index + 1} dalam sinyal')
    plt.legend()

    # Menampilkan hasil windowing untuk frame ke-100
    plt.subplot(2, 1, 2)
    plt.plot(frame_to_show, label=f'Frame {frame_index + 1}')  # Menggunakan frame_index
    plt.title(f'hasil windowing untuk frame ke- {frame_index + 1}')
    plt.tight_layout()
    
    # Simpan gambar framing dan windowing
    image_path = os.path.join('static', 'framing_plot.png')
    plt.savefig(image_path, format='png')
    return frames

def fouriertransform(mp):
	NFFT = 512
	frames = framing(mp)
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
	pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
	return pow_frames

def filterbanks(mp):
  nfilt = 40
  low_freq_mel = 0
  NFFT = 512

  sr, signal = initialize(mp)
  high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
  mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
  hz_points = (700 * (10**(mel_points / 2595) - 1))
  bin = np.floor((NFFT + 1) * hz_points / sr)

  pow_frames = fouriertransform(mp)
  fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
  for m in range(1, nfilt + 1):
      f_m_minus = int(bin[m - 1])
      f_m = int(bin[m])
      f_m_plus = int(bin[m + 1])

      for k in range(f_m_minus, f_m):
          fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
      for k in range(f_m, f_m_plus):
          fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
  filter_banks = np.dot(pow_frames, fbank.T)
  filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
  filter_banks = 20 * np.log10(filter_banks)

  plt.figure(figsize=(8, 6))
  for i in range(len(filter_banks)):
    plt.plot(filter_banks[i])
  plt.title("Filter Bank")
  plt.xlabel("Frequency Bins")
  plt.ylabel("Amplitude")
  filterbank_image_path = os.path.join('static', 'filterbank_plot.png')
  plt.savefig(filterbank_image_path, format='png')


  return filter_banks
    
def mfcct(mp):
    cep_lifter = 22
    filter_banks = filterbanks(mp)

    # Hitung MFCC sebelum penggunaan cep_lifter
    mfcc_before = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc_before.shape
    mfcc_before = np.mean(mfcc_before, axis=0)

    # Simpan gambar hasil DCT sebelum cep_lifter sebagai berkas gambar
    plt.figure(figsize=(8, 6))
    plt.plot(mfcc_before)
    plt.title("DCT Result Before cep_lifter")
    plt.xlabel("Coefficients")
    plt.ylabel("Amplitude")
    dct_image_before_path = os.path.join('static', 'dct_before_plot.png')
    plt.savefig(dct_image_before_path, format='png')

    # Hitung MFCC setelah penggunaan cep_lifter
    mfcc_after = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]

    (nframes, ncoeff) = mfcc_after.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc_after *= lift
    mfcc_after = np.mean(mfcc_after, axis=0)

    # Simpan gambar hasil DCT setelah cep_lifter sebagai berkas gambar
    plt.figure(figsize=(8, 6))
    plt.plot(mfcc_after)
    plt.title("DCT Result After cep_lifter")
    plt.xlabel("Coefficients")
    plt.ylabel("Amplitude")
    dct_image_after_path = os.path.join('static', 'dct_after_plot.png')
    plt.savefig(dct_image_after_path, format='png')

    return mfcc_before, mfcc_after


# def	mfcct(mp):
#   cep_lifter = 22
#   filter_banks = filterbanks(mp)
#   mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
#   (nframes, ncoeff) = mfcc.shape
#   n = np.arange(ncoeff)
#   lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
#   mfcc *= lift
#   mfcc = np.mean(mfcc, axis=0)

#     # Simpan gambar hasil DCT sebagai berkas gambar
#   plt.figure(figsize=(8, 6))
#   plt.plot(mfcc)
#   plt.title("DCT Result")
#   plt.xlabel("Coefficients")
#   plt.ylabel("Amplitude")
#   dct_image_path = os.path.join('static', 'dct_plot.png')
#   plt.savefig(dct_image_path, format='png')

#   return mfcc

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    result = None
    distances = []
    if request.method == 'POST':
        file = request.files['file']
        unique_filename = str(uuid.uuid4()) + '.wav'
        file.save(unique_filename)
        mp = unique_filename
        mfcc_before, mfcc_after = mfcct(mp)
        mfcc_features = mfcc_after.reshape(1, -1)
        prediction = knn_model.predict(mfcc_features)
        filter_banks = filterbanks(mp)
        filterbank_table = filter_banks.tolist()

        # Hitung jarak Euclidean antara data baru dan data pelatihan
        for i, train_data_point in enumerate(train_data):
            # Ambil fitur MFCC dari train_data_point
            train_mfcc = train_data_point['mfcc']
            
            # Misalnya, hitung jarak Euclidean antara mfcc_features dan train_mfcc
            distance = np.linalg.norm(mfcc_features - train_mfcc)
            distances.append((i, distance))
        # Urutkan jarak-jarak tersebut berdasarkan jarak
        distances.sort(key=lambda x: x[1])

        # Pilih 5 data terdekat
        k = 7
        nearest_neighbors = distances[:k]

        # Ambil kelas data terdekat
        predicted_classes = [train_data[i]['label'] for i, _ in nearest_neighbors]

        if prediction[0] == 'belly_pain':
            result = "Belly pain"
            pre_emphasis_image = 'pre_emphasis.png'
            framing_image = 'framing_plot.png'
            spectrum_image = 'spectrum_plot.png'
            filterbank_image= 'filterbank_plot.png'
            mfcc_before_image= 'dct_before_plot.png'
            mfcc_after_image= 'dct_after_plot.png'
            # dct_image= 'dct_plot.png'
            
        elif 'burping' in prediction :
            result = "Burping"
            pre_emphasis_image = 'pre_emphasis.png'
            framing_image = 'framing_plot.png'
            spectrum_image = 'spectrum_plot.png'
            filterbank_image= 'filterbank_plot.png'
            mfcc_before_image= 'dct_before_plot.png'
            mfcc_after_image= 'dct_after_plot.png'
            # dct_image= 'dct_plot.png'
            
        elif 'discomfort' in prediction :
            result = "Discomfort"
            pre_emphasis_image = 'pre_emphasis.png'
            framing_image = 'framing_plot.png'
            spectrum_image = 'spectrum_plot.png'
            filterbank_image= 'filterbank_plot.png'
            mfcc_before_image= 'dct_before_plot.png'
            mfcc_after_image= 'dct_after_plot.png'
            # dct_image= 'dct_plot.png'
            
        elif 'hungry' in prediction :
            result = "Hungry"
            pre_emphasis_image = 'pre_emphasis.png'
            framing_image = 'framing_plot.png'
            spectrum_image = 'spectrum_plot.png'
            filterbank_image= 'filterbank_plot.png'
            mfcc_before_image= 'dct_before_plot.png'
            mfcc_after_image= 'dct_after_plot.png'
            # dct_image= 'dct_plot.png'
            
        elif 'tired' in prediction :
            result = "Tired"
            pre_emphasis_image = 'pre_emphasis.png'
            framing_image = 'framing_plot.png'
            spectrum_image = 'spectrum_plot.png'
            filterbank_image= 'filterbank_plot.png'
            mfcc_before_image= 'dct_before_plot.png'
            mfcc_after_image= 'dct_after_plot.png'
            # dct_image= 'dct_plot.png'
        else :
            result = "errorrrrrrrrrrrr"

        # Menambahkan visualisasi spektrum
        sr, signal = initialize(mp)
        plt.figure(figsize=(12, 4))
        plt.specgram(signal, NFFT=512, Fs=sr, noverlap=256, cmap='viridis')
        plt.title('Spectrogram of the Audio Signal')
        spectrum_image_path = os.path.join('static', 'spectrum_plot.png')
        plt.savefig(spectrum_image_path, format='png')


    return render_template(
        'home.html',
        result=result,
        pre_emphasis_image=pre_emphasis_image,
        framing_image=framing_image,
        spectrum_image=spectrum_image,
        filterbank_image=filterbank_image,
        mfcc_before_image=mfcc_before_image,
        mfcc_after_image=mfcc_after_image,
        mfcc=mfcc_features[0],
        predicted_classes=predicted_classes,
        euclidean_distances=[distance for _, distance in nearest_neighbors],
        filterbank_table=filterbank_table
        # mfcc_before=mfcc_before_str
    )


if __name__ == '__main__':
    app.run(debug=True)
