Title:  TUT Sound events 2016, Evaluation dataset

TUT Sound events 2016, Evaluation dataset
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
TUT Sound events 2016, evaluation dataset consists of 10 audio recordings from two acoustic scenes: 

- Home (indoor), 5 recordings, totaling 17:49
- Residential area (outdoor), 5 recordings, totaling 17:49

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
│   README.md						This file, markdown-format
│   README.html						This file, html-format
│   EULA.pdf						End user license agreement
│   meta.txt						Meta data, csv-format, [audio file][tab][scene label][tab][event onset][tab][event offset][tab][event label][tab][event type]
│
└───audio							10 audio files, 24-bit 44.1kHz
│   │
│   └───home						Acoustic scene label
│   │   │   a029.wav				Name format: [original_recording_identifier].wav
│   │   │   a033.wav
│   │       ...
│   └───residential_area	
│       │   a008.wav		
│       │   a009.wav
│           ...
│
└───evaluation_setup				Cross-validation setup, 4 folds
│   │   home_test.txt 				Testing file list, csv-format: [audio file (string)][tab][scene label (string)]
│   │   home_fold1_evaluate.txt 	Evaluation file list, fold1_test.txt with added ground truth, csv-format: [audio file (string)][tab][scene label (string)][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]
│   │   ...        
│
└───meta							Meta data, only with target sound classes
│   │
│   └───home						Acoustic scene label
│   │   │   a029.ann				Annotation data, csv-format (can be imported to audacity): [event onset (float)][tab][event offset (float)][tab][event label (string)]
│   │   │   a033.ann
│   │       ...
│   └───residential_area	
│       │   a008.ann		
│       │   a009.ann
│           ...
└───meta_full						Meta data, all with all sound classes
    │
    └───home						Acoustic scene label
    │   │   a029_full.ann			Full annotation data, csv-format (can be imported to audacity): [event onset (float)][tab][event offset (float)][tab][event label (string)]
    │   │   a031_full.ann
    │       ...
    └───residential_area	
        │   a008_full.ann		
        │   a009_full.ann    
        	...
```

2. Usage
=================================

Partitioning of data into **development dataset** and **evaluation dataset** was done based on the amount of examples available for each event class, while also taking into account recording location. Ideally the subsets should have the same amount of data for each class, or at least the same relative amount, such as a 70-30% split. Because the event instances belonging to different classes are distributed unevenly within the recordings, the partitioning of individual classes can be controlled only to a certain extent. 

The split condition was relaxed from 70-30%. For home, 40-80% of instances of each class were selected into the development set. For residential area, 60-80% of instances of each class were selected into the development set.  

Development dataset is provided separately.

### Evaluation setup

The **development dataset** should be used as training set, and **evaluation dataset** should be used for testing. The setup is provided with the dataset in the directory `evaluation_setup`. 

#### Testing

`evaluation setup\[scene_label]_test.txt`
: testing file list (in csv-format)

Format:

    [audio file (string)][tab][scene label (string)]
 
#### Evaluating

`evaluation setup\[scene_label]_evaluate.txt`
: evaluation file list (in csv-format), same as [scene_label]_test.txt but added with ground truth information. These two files are provided separately to prevent contamination with ground truth when testing the system. 

Format: 

    [audio file (string)][tab][scene label][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]

3. Changelog
=================================
#### 1.0 / 2016-05-31
* Initial commit for DCASE2016 challenge
#### 1.1 / 2017-09-25
* Meta data added

4. License
=================================

See file [EULA.pdf](EULA.pdf)