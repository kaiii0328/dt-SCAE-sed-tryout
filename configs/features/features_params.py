sample_rate = 16000

params = {
    'fs': sample_rate,
    'win_length_samples': int(0.040 * sample_rate),
    'hop_length_samples': int(0.020 * sample_rate),
    'mono': True,  # [True, False]
    'window': 'hamming_asymmetric',  # [hann_asymmetric, hamming_asymmetric]
    'n_fft': 1024,  # FFT length
    'spectrogram_type': 'magnitude',  # [magnitude, power]
    'compute_mels': True,  # compute spectrogram or MELs
    'n_mels': 40,  # Number of MEL bands used
    'normalize_mel_bands': False,  # [True, False]
    'fmin': 0,  # Minimum frequency when constructing MEL bands
    'fmax': 8000,  # Maximum frequency when constructing MEL band
    'htk': False,  # Switch for HTK-styled MEL-frequency equation
    'log': True,  # Logarithmic
    'delta_width': 0,  # delta_width
    'pad_in_audio': None,  # len in seconds to pad each sequence, else None
    'normalize_audio': True
}
