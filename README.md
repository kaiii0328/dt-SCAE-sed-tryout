# Polyphonic Sound Event Detection by using Capsule Neural Networks

This is the source code for the system described in the paper [Polyphonic Sound Event Detection by using Capsule Neural Networks](https://arxiv.org/pdf/1810.06325.pdf).

Requirements
This software requires Python 3.5 or higher. It is recommended to create a Python virtual environment to isolate package installation from the system.

To install the dependencies, run:

    pip install -r requirements.txt

The main functionality of this software also requires the DCASE 2016 Task3 and DCASE 2017 Task 3 datasets, which may be downloaded here:

[DCASE-2016](http://www.cs.tut.fi/sgn/arg/dcase2016/task-sound-event-detection-in-real-life-audio),
[DCASE-2017](http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-sound-event-detection-in-real-life-audio)

## Usage

Using this software, the user is able to extract feature vectors, train the network, generate predictions using the trained network, and evaluate the prediction
both on the development and the evaluation sets of the respective datasets.

### **Features Extraction**



To extract feature vectors, run:

    python 1_features_extraction_dcase.py -i path/to/DCASE2017-dataset/../audio/street/ -o ./dataset/TUT-sound-events-2017-development/features/logmel_40_mbe

See:

    configs/features/features_params.py


for tweaking the parameters. By default it computes binaural STFT spectrograms from the input audio files.
Please be sure to process all the wav files componing the dataset


### **Cross-Validation and Evaluation**

To execute an experiment, run:

    python 2_main_experiment_DCASE.py -cf configs/neural_networks/Capsule_DCASE_2017.cfg

See:

    configs/neural_networks/Capsule_DCASE_2017.cfg 


for tweaking the parameters. You may also want to change the work path:

    --root-path          ./                   """Path to parent directory"""

Please notice that you will have to modify --dataset-list-filepath, --evaluation-list-filepath, --dataset-path, --evaluation-set-path accordingly.

### **About Results Reproducibility**

We recommend to run the experiments more than one time in order to reproduce the results we reported in the paper.
GPUs were used to train the models, and although we fixed the seeds of the random generators we cannot assure the exact reproduction of the results for each run.
