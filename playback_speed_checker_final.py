# -*- coding: utf-8 -*-
"""
@work: SMC Master Thesis UPF 2016
@author: Ignasi Adell Arteaga
"""


from __future__ import unicode_literals

import os
import sys
import numpy
import soundfile as sf
import mlpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.dataset import Instance
from weka.classifiers import Classifier

#import sys  
#reload(sys)  
#sys.setdefaultencoding("ISO-8859-1")


def dtw_checker(inputFile1, inputFile2):
    ''' COMPARES TO GIVEN FILES '''

    # DEFINING CONSTANTS
    DIST_STEP = 2000

    (inAudio1, fs1) = file_reader(inputFile1)
    (inAudio2, fs2) = file_reader(inputFile2)

    # APPLY DTW ALGORITHM (http://mlpy.sourceforge.net/docs/3.5/dtw.html)   
    dist_array = []
    i = 1

    while (i < len(inAudio1)):
        (dist, cost, path) = mlpy.dtw_std(inAudio1[i:DIST_STEP+i], inAudio2[i:DIST_STEP+i], dist_only=False)
        dist_array.append(dist)
        i+=DIST_STEP

    dist_array = numpy.asarray(dist_array)
    dist_final = numpy.mean(dist_array)

    result = os.path.basename(inputFile1) + " -- " + os.path.basename(inputFile2) + " ==> " + str(dist_final)
#    print result

    return (result, dist_final)


def file_reader(inputFile):
    ''' READS GIVEN FILE AND CONVERTS IT TO MONO. '''

    # DEFINING CONSTANTS
    PROCESSING_WINDOW = 15              # Processing time window (in seconds).    

    # READ STEREO AUDIO FILE
    (inAudio, fs) = sf.read(inputFile)

    # TAKE CENTRAL PART OF FILE FOR ANALYSIS
    half_file = float(inAudio.size/4)   # 2 channels, divide by 2 twice
    first_half = int(round(half_file - PROCESSING_WINDOW/2*fs))
    second_half = int(round(half_file + PROCESSING_WINDOW/2*fs))
    inAudio = inAudio[first_half:second_half, :]

    # EACH CHANNEL WITHOUT NORMALIZING
    in_chan1_raw = inAudio[:,0].astype(float)
    in_chan2_raw = inAudio[:,1].astype(float)

    # Convert to mono RAW
    in_mono_raw = (in_chan1_raw + in_chan2_raw)/2
    in_mono_raw = numpy.array(in_mono_raw)

    return (in_mono_raw, fs)


def playback_speed_checker(inputFile, dirRef):
    
    TRAINING_ARFF = 'dataset_playback.arff'
    inputRef = ""

    # Start JVM
    jvm.start()
    jvm.start(system_cp=True, packages=True)
    jvm.start(max_heap_size="512m")
    
    # Find reference file
    for file in os.listdir(dirRef):
        if str(file).find(str(os.path.basename(inputFile))) != -1:
            inputRef = os.path.join(dirRef, file)
            break

    # Calculation distance
    (result, distance) = dtw_checker(inputFile, inputRef)

    # Loading data
    loader = Loader(classname="weka.core.converters.ArffLoader")    
    data = loader.load_file(TRAINING_ARFF)
    data.class_is_last()                    # set class attribute

    # Train the classifier
    #cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls = Classifier(classname="weka.classifiers.trees.J48", options = ["-C", "0.3", "-M", "10"])
    cls.build_classifier(data)

    # Classify instance
    speed_instance = Instance.create_instance(numpy.ndarray(distance), classname='weka.core.DenseInstance', weight=1.0)
    speed_instance.dataset = data
    
    # Classify instance
    speed_flag = cls.classify_instance(speed_instance)
    
    if (distance == 0):
        speed_class = 'nominal'
    else:
        if speed_flag == 0: speed_class = 'down_speed'
        if speed_flag == 0: speed_class = 'up_speed'
        
#    print os.path.basename(inputFile) + ' --- ' + speed_class
    
    # Stop JVM
    jvm.stop()    

    print "SPEED IS: " + speed_class

    return speed_class


def speed_distance_evaluation(dirRef, dirTest):
    
    dataset_file = os.path.join(dirRef, 'dataset_playback.arff')    
    f = open(dataset_file, 'w')

    f.writelines('@RELATION playback_speed\n')
    f.writelines('@ATTRIBUTE distance NUMERIC\n')   
    f.writelines('@ATTRIBUTE class {speed_minus1, speed_plus1, speed_minus2, speed_plus2, speed_minus5, speed_plus5, speed_minus8, speed_plus8}\n')
    f.writelines('@DATA\n')
    
    for file1 in os.listdir(dirRef):
        if file1.endswith(".wav"):   
            for file2 in os.listdir(dirTest):
                if file2.endswith(str(file1)):
                    filename = os.path.join(dirTest, file2)
                    (result, dist) = dtw_checker(os.path.join(dirRef, file1), os.path.join(dirTest, file2))

                    inputFile = filename
        
                    if (inputFile.find('speed_minus1_') != -1): riaa_class = 'speed_minus1'
                    if (inputFile.find('speed_minus2_') != -1): riaa_class = 'speed_minus2'
                    if (inputFile.find('speed_minus5_') != -1): riaa_class = 'speed_minus5'
                    if (inputFile.find('speed_minus8_') != -1): riaa_class = 'speed_minus8'

                    if (inputFile.find('speed_plus1_') != -1): riaa_class = 'speed_plus1'
                    if (inputFile.find('speed_plus2_') != -1): riaa_class = 'speed_plus2'
                    if (inputFile.find('speed_plus5_') != -1): riaa_class = 'speed_plus5'
                    if (inputFile.find('speed_plus8_') != -1): riaa_class = 'speed_plus8'
                    
                    instance_tuple = (dist, riaa_class)      
                    instance_tuple = str(instance_tuple)
        #            instance_tuple.replace("[", "")
        #            instance_tuple.replace("]", "")
        #            instance_tuple.replace("'", "")
                   
                    f.writelines(instance_tuple)
                    f.writelines ('\n')
    
    f.close
    
    return (dist, riaa_class)


def batch_processing_and_boxplot(dirRef, dirTest):
    ''' RUNS BATCH PROCESSING OF FILES IN A DIR'''

    playback_file_abs = os.path.join(dirRef, 'result_abs.txt')   
    f = open(playback_file_abs, 'w')

    dist_vectors_array = []    

    for file1 in os.listdir(dirRef):
        if file1.endswith(".wav"):

            dist_vector = []            

            for file2 in os.listdir(dirTest):
                if file2.endswith(str(file1)):
                    (result, dist) = dtw_checker(os.path.join(dirRef, file1), os.path.join(dirTest, file2))
                    f.writelines(result + '\n')
                    dist_vector.append(dist)

            dist_vectors_array.append(numpy.asarray(dist_vector))

    f.close  

    # BOX PLOT REPRESENTATION
    dist_vectors_array = numpy.asarray(dist_vectors_array)
    plt.figure()
    plt.title('DTW Distances')
    plt.ylabel('Distance value')
    plt.xlabel('Speed variation [%]')
    plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=0.7)
    plt.boxplot(dist_vectors_array)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['-1%', '-2%', '-5%', '-8%', '+1%', 
                   '+2%', '+5%', '+8%'], rotation=45)
    axes = plt.gca()
    axes.yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.savefig(os.path.join(dirRef, 'dtw_distances.jpg'))

    return 0
    

if __name__=='__main__':
    playback_speed_checker(sys.argv[1], sys.argv[2])

