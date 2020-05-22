#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import multiprocessing
from tqdm import tqdm
import librosa
import scipy.signal as spy
# import soundfile as sf
import sys
import errno
import argparse
from configs.features.features_params import *


def mel_compute(files, force, infold, outfold):


    """Mel-band energies

    Parameters
    ----------
    data : numpy.ndarray
        Audio data
    params : dict
        Parameters

    Returns
    -------
    list of numpy.ndarrays
        List of feature matrices, feature matrix per audio channel

    """
    for file in tqdm(files):
        # use the name of the file without the extension
        filename, ext = os.path.splitext(file)
        filename_split = filename.split(os.sep)
        # THIS IS CUSTOM, based on dataset folder tree
        filename = filename_split[-1]
        # the .npy file name
        print('')
        print("Processing: " + file)
        npy_file = os.path.join(outfold, filename)
        # check if the file already exists
        if not force and os.path.exists(npy_file):
            print('file', npy_file, 'already exists, use -f to force overwrite')
            input()
        # open the wav file
        # data, samplerate = sf.read(file)
        data, samplerate = librosa.core.load(file, sr=params['fs'], mono=params['mono'])
        # check file lenght
        if params['mono']:
            file_length = len(data)
            data = np.reshape(data, (1, -1))
        else:
            file_length = data.shape[1]

        if params['pad_in_audio'] is not None:
            if file_length / samplerate > params['pad_in_audio']:
                print('Truncating lenght exceeding audiofile')
                data = data[:params['pad_in_audio']*samplerate]
            elif file_length / samplerate < params['pad_in_audio']:
                print('Padding lenght exceeding audiofile')
                sample_to_add = (params['pad_in_audio']*samplerate) - len(data)
                noise = np.random.normal(0, 0.00001, size=sample_to_add)
                # noise = np.zeros((sample_to_add, 1))
                data = np.concatenate((data, noise))
            else:
                print('Length audiofile in sec. is equal to: '+str(params['pad_in_audio']))

        eps = np.spacing(1)
        if params['normalize_audio']:
            # normalize audio
            data = data + eps
            head_room = 0.005
            mean_value = np.mean(data)
            data -= mean_value
            max_value = np.max(abs(data)) + head_room
            data = data / max_value

        window = spy.hamming(params['win_length_samples'], sym=False)

        for channel in range(0, data.shape[0]):
            spectrogram = np.abs(librosa.stft(data[channel,:],
                                              n_fft=params['n_fft'],
                                              win_length=params['win_length_samples'],
                                              hop_length=params['hop_length_samples'],
                                              center=True,
                                              window=window))
            spectrogram = np.asarray(spectrogram)

            if params['compute_mels']:
                mel_basis = librosa.filters.mel(sr=params['fs'],
                                                n_fft=params['n_fft'],
                                                n_mels=params['n_mels'],
                                                fmin=params['fmin'],
                                                fmax=params['fmax'],
                                                htk=params['htk'])
                mel_basis = np.asarray(mel_basis)
                if params['normalize_mel_bands']:
                    mel_basis /= np.max(mel_basis, axis=-1)[:, None]

                mel_spectrum = np.dot(mel_basis, spectrogram)
            else:
                mel_spectrum = spectrogram

            if params['log']:
                mel_spectrum = np.log(mel_spectrum + eps)
                mel_spectrum = mel_spectrum

            # add delta e delta-deltas
            if params['delta_width'] != 0:
                deltas = librosa.feature.delta(mel_spectrum, params['delta_width'] * 2 + 1, order=1, axis=-1)
                # acc = librosa.feature.delta(mel_spectrum, params['delta_width'] * 2 + 1, order=2, axis=-1)
                mel_spectrum = np.stack((mel_spectrum, deltas), axis=2)
                # mel_spectrum = np.concatenate((mel_spectrum, acc), axis=1)
            if params['mono'] is False:
                if 'feature_matrix' in locals():
                    feature_matrix = np.stack((feature_matrix, mel_spectrum), axis=2)
                else:
                    feature_matrix = np.array(mel_spectrum, dtype=np.float32)  # shape = features, time, channels
            else:
                feature_matrix = np.array(mel_spectrum, dtype=np.float32)  # shape = features, time
        feature_matrix = feature_matrix.astype(np.float32)
        np.save(npy_file, feature_matrix)  # shape = features, time

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.pcolormesh(spectrogram)
        # plt.show()


def chunk(lst, n):
    return [lst[i::n] for i in range(n)]


def main():
    # parse arguments

    parser = argparse.ArgumentParser(description='Logmel features extraction script')
    parser.add_argument('-i', dest='infile', default=None)
    parser.add_argument('-o', dest='outfile', default=None)
    parser.add_argument('-m', action='store_true', dest='multiproc', default=False)
    parser.add_argument('-f', action='store_true', dest='force', default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.infile:
        if args.outfile is None:
            args.outfile = os.path.join(args.infile, 'features', 'logmel_40_bin_mbe')

        # generate wavfiles list
        wavfiles = []
        for (dirpath, dirnames, filenames) in os.walk(args.infile):
            # drop all non wav file
            for f in filenames:
                if f.lower().endswith('.wav'):
                    wavfile = os.path.join(dirpath, f)
                    wavfiles.append(wavfile)

        try:
            os.makedirs(args.outfile)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
            pass

        if args.multiproc is False:
            for wavfile in tqdm(wavfiles):
                wavfile = [wavfile]
                mel_compute(wavfile, args.force, args.infile, args.outfile)
        else:
            jobs = []
            numCPU = multiprocessing.cpu_count()
            numCPU = numCPU
            # file division
            _files = chunk(wavfiles, numCPU)
            # print len(_files)
            # raw_input()

            # check
            _sum = 0
            for i in range(len(_files)):
                _sum += len(_files[i])

            if not (_sum == len(wavfiles)):
                print('ERROR: some files are missed.')
                sys.exit()

            for i in range(len(_files)):
                p = multiprocessing.Process(target=mel_compute, args=(_files[i], args.force, args.infile, args.outfile))
                jobs.append(p)
                p.start()

            # Wait for all worker processes to finish
            for p in jobs:
                p.join()

        print('...done')

    else:

        print('Please specify wav files directory (-i)')


if __name__ == '__main__':
    main()
