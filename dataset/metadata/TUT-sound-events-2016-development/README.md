Title:  TUT Sound events 2016, Development dataset

TUT Sound events 2016, Development dataset
==========================================
[Audio Research Group / Tampere University of Technology](http://arg.cs.tut.fi/)

Authors
- Toni Heittola (<toni.heittola@tut.fi>, <http://www.cs.tut.fi/~heittolt/>)
- Annamaria Mesaros (<annamaria.mesaros@tut.fi>, <http://www.cs.tut.fi/~mesaros/>)
- Tuomas Virtanen (<tuomas.virtanen@tut.fi>, <http://www.cs.tut.fi/~tuomasv/>)

Recording and annotation
- Eemi Fagerlund
- Aku Hiltunen

# Table of Contents
1. [Dataset](#1-dataset)
2. [Usage](#2-usage)
3. [Changelog](#3-changelog)
4. [License](#4-license)

1. Dataset
=================================
TUT Sound events 2016, development dataset consists of 22 audio recordings from two acoustic scenes: 

- Home (indoor), 10 recordings, totaling 36:16
- Residential area (outdoor), 12 recordings, totalling 42:00

The dataset was collected in Finland by Tampere University of Technology between 06/2015 - 01/2016. The data collection has received funding from the European Research Council under the ERC Grant Agreement 637422 EVERYSOUND.

[![ERC](https://erc.europa.eu/sites/default/files/content/erc_banner-horizontal.jpg "ERC")](https://erc.europa.eu/)

### Preparation of the dataset

The recordings were captured each in a different location: different streets, different homes. The equipment used for recording consists of a binaural [Soundman OKM II Klassik/studio A3](http://www.soundman.de/en/products/) electret in-ear microphone and a [Roland Edirol R-09](http://www.rolandus.com/products/r-09/) wave recorder using 44.1 kHz sampling rate and 24 bit resolution. 

For audio material recorded in private places, written consent was obtained from all people involved. Material recorded in public places (residential area) does not require such consent.

Individual sound events in each recording were annotated by two research assistants using freely chosen labels for sounds. Annotators were trained first on few example recordings. They were instructed to annotate all audible sound events, and choose event labels freely. This resulted in a large set of raw labels. *Target sound event* classes for the dataset were selected based on the frequency of the obtained labels, to ensure that the selected sounds are common for an acoustic scene, and there are sufficient examples for learning acoustic models. 

The ground truth is provided as a list of the sound events present in the recording, with annotated onset and offset for each sound instance. Annotations with only targeted sound events classes are in the directory `meta`, and annotations with full set of annotated sound events are in the directory `meta_full`. Only targeted sound events should be evaluated.

### File structure

```
dataset root
│   README.md				this file, markdown-format
│   README.html				this file, html-format
│   EULA.pdf				End user license agreement
│   meta.txt				meta data, csv-format, [audio file][tab][scene label][tab][event onset][tab][event offset][tab][event label][tab][event type]
│
└───audio					22 audio files, 24-bit 44.1kHz
│   │
│   └───home				acoustic scene label
│   │   │   a030.wav		name format: [original_recording_identifier].wav
│   │   │   a031.wav
│   │       ...
│   └───residential_area	
│       │   a001.wav		
│       │   a002.wav
│           ...
│
└───evaluation_setup		cross-validation setup, 4 folds
│   │   home_fold1_train.txt		training file list, csv-format: [audio file (string)][tab][scene label (string)][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]
│   │   home_fold1_test.txt 		testing file list, csv-format: [audio file (string)][tab][scene label (string)]
│   │   home_fold1_evaluate.txt 	evaluation file list, fold1_test.txt with added ground truth, csv-format: [audio file (string)][tab][scene label (string)][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]
│   │   ...        
│
└───meta					meta data, only with target sound classes
│   │
│   └───home				acoustic scene label
│   │   │   a030.ann		annotation data, csv-format (can be imported to audacity): [event onset (float)][tab][event offset (float)][tab][event label (string)]
│   │   │   a031.ann
│   │       ...
│   └───residential_area	
│       │   a001.ann		
│       │   a002.ann
│           ...
└───meta_full				meta data, all with all sound classes
    │
    └───home				acoustic scene label
    │   │   a030_full.ann	full annotation data, csv-format (can be imported to audacity): [event onset (float)][tab][event offset (float)][tab][event label (string)]
    │   │   a031_full.ann
    │       ...
    └───residential_area	
        │   a001_full.ann		
        │   a002_full.ann    
        	...
```

### Event statistics

The sound event instance counts for the dataset are shown below. It can be observed that in residential area scenes, the sound event classes are mostly related to concrete physical sound sources - bird singing, car passing by - while the home scenes are dominated by abstract object impact sounds, besides some more well defined (still impact) dishes, cutlery, etc.

** Home **

| Event label           | Event count |
|-----------------------|-------------|
| (object) rustling     | 41          |
| (object) snapping     | 42          |
| cupboard              | 27          |
| cutlery               | 56          |
| dishes                | 94          |
| drawer                | 23          |
| glass jingling        | 26          |
| object impact         | 155         |
| people walking        | 24          |
| washing dishes        | 60          |
| water tap running     | 37          |
| **Total**             | **585**     |

** Residential area **

| Event label           | Event count |
|-----------------------|-------------|
| (object) banging      | 15          |
| bird singing          | 162         |
| car passing by        | 74          |
| children shouting     | 23          |
| people speaking       | 41          |
| people walking        | 32          |
| wind blowing          | 22          |
| **Total**             | **369**     |

2. Usage
=================================

Partitioning of data into **development dataset** and **evaluation dataset** was done based on the amount of examples available for each event class, while also taking into account recording location. Ideally the subsets should have the same amount of data for each class, or at least the same relative amount, such as a 70-30% split. Because the event instances belonging to different classes are distributed unevenly within the recordings, the partitioning of individual classes can be controlled only to a certain extent. 

The split condition was relaxed from 70-30%. For home, 40-80% of instances of each class were selected into the development set. For residential area, 60-80% of instances of each class were selected into the development set.  

Evaluation dataset is provided separately.

### Cross-validation setup

A cross-validation setup is provided in order to make results reported with this dataset uniform. The setup consists of four folds, so that each recording is used exactly once as test data. At this stage the only condition imposed was that the test subset does not contain classes unavailable in training. 

The folds are provided with the dataset in the directory `evaluation_setup`. 

#### Training

`evaluation setup\[scene_label]_fold[1-4]_train.txt`
: training file list (in csv-format)

Format:

    [audio file (string)][tab][scene label (string)][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]

#### Testing

`evaluation setup\[scene_label]_fold[1-4]_test.txt`
: testing file list (in csv-format)

Format:

    [audio file (string)][tab][scene label (string)]

#### Evaluating

`evaluation setup\[scene_label]_fold[1-4]_evaluate.txt`
: evaluation file list (in csv-format), same as fold[1-4]_test.txt but added with ground truth information. These two files are provided separately to prevent contamination with ground truth when testing the system. 

Format: 

    [audio file (string)][tab][scene label][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]
 
3. Changelog
=================================
#### 1.0 / 2016-02-05
* Initial commit

4. License
=================================

See file [EULA.pdf](EULA.pdf)