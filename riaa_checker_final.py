# -*- coding: utf-8 -*-
"""
@work: SMC Master Thesis UPF 2016
@author: Ignasi Adell Arteaga
"""


import os
import matlab.engine
import numpy
import sys

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.dataset import Instance
from weka.classifiers import Classifier

import matplotlib as mpl
import matplotlib.ticker as ticker
#mpl.use('Gtk3Agg') # I'd advise testing 'Gtk3Agg' or 'Qt4Agg' (or 5) instead
import matplotlib.pyplot as plt



def riaa_checker(inputFile):
    
    TRAINING_ARFF = 'C:\Users\ASUS\Desktop\IGNASI\SMC\Workspace\dataset_riaa.arff'

    # Start JVM
    jvm.start()
    jvm.start(system_cp=True, packages=True)
    jvm.start(max_heap_size="512m")

    # Calculation of bark bands information
    (absolute_bark, relative_bark, bark_ratios) = compute_bark_spectrum(inputFile)

    # Loading data
    loader = Loader(classname="weka.core.converters.ArffLoader")    
    data = loader.load_file(TRAINING_ARFF)
    data.class_is_last()                    # set class attribute

    # Train the classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    #cls = Classifier(classname="weka.classifiers.trees.J48", options = ["-C", "0.3", "-M", "10"])
    cls.build_classifier(data)

    # Classify instance
    bark_instance = Instance.create_instance(bark_ratios, classname='weka.core.DenseInstance', weight=1.0)
    bark_instance.dataset = data
    
    # Classify instance
    riaa_flag = cls.classify_instance(bark_instance)
    
    if riaa_flag == 0:
        riaa_class = 'riaa_ok'
    else:
        riaa_class = 'riaa_ko'
        
#    print os.path.basename(inputFile) + ' --- ' + riaa_class
    
    # Stop JVM
    jvm.stop()   

    print "RIAA FILTERING?: " + riaa_class

    return riaa_class
    

def batch_riaa_checking(inputDir):

    # Start JVM
    jvm.start()
    jvm.start(system_cp=True, packages=True)
    jvm.start(max_heap_size="512m")

    riaa_ok = 0
    riaa_ko = 0

    for file in os.listdir(inputDir):
        if file.endswith(".wav"):
            riaa_flag = riaa_checker(os.path.join(inputDir, file))
            if (riaa_flag == 'riaa_ko'): riaa_ko+=1
            if (riaa_flag == 'riaa_ok'): riaa_ok+=1
    
    # Stop JVM
    jvm.stop()      
    
    return (riaa_ko, riaa_ok)


def compute_bark_spectrum(inputFile):
    
    file_dir = os.path.dirname(inputFile)

    # Run matlab engine for the MA-Toolbox
    eng = matlab.engine.start_matlab()
    eng.addpath(file_dir, nargout=0)
    
    # Calculate both absolute and relative bark representations
    (bark_spectrum, relative_bark) = eng.bark_spectrum_processing(inputFile, nargout=2)

    ratios_list = []

    # Calculate rations against central bark (1000Hz, bark9)
    ratios_list.append(numpy.float32(relative_bark[0])/numpy.float32(relative_bark[8]))     # bark1 against bark9
    ratios_list.append(numpy.float32(relative_bark[1])/numpy.float32(relative_bark[8]))     # bark2 against bark9
    ratios_list.append(numpy.float32(relative_bark[2])/numpy.float32(relative_bark[8]))     # bark3 against bark9
    ratios_list.append(numpy.float32(relative_bark[21])/numpy.float32(relative_bark[8]))    # bark22 against bark9
    ratios_list.append(numpy.float32(relative_bark[22])/numpy.float32(relative_bark[8]))    # bark23 against bark9
    ratios_list.append(numpy.float32(relative_bark[23])/numpy.float32(relative_bark[8]))    # bark24 against bark9

    ratios_list.append(numpy.float32(relative_bark[3])/numpy.float32(relative_bark[0]))    # bark4 against bark1
    ratios_list.append(numpy.float32(relative_bark[4])/numpy.float32(relative_bark[0]))    # bark5 against bark1
    ratios_list.append(numpy.float32(relative_bark[5])/numpy.float32(relative_bark[0]))    # bark6 against bark1   
    
#    bark_ratios = (ratio_1_9, ratio_2_9, ratio_3_9, ratio_22_9, ratio_23_9, ratio_24_9, ratio_4_1, ratio_5_1, ratio_6_1)

    return (bark_spectrum, relative_bark, ratios_list)
   

def bark_ratio_evaluation(inputDir):
    
    dataset_file = os.path.join(inputDir, 'dataset.arff')    
    f = open(dataset_file, 'w')

    f.writelines('@RELATION riaa\n')
    f.writelines('@ATTRIBUTE ratio_1_9 NUMERIC\n')
    f.writelines('@ATTRIBUTE ratio_2_9 NUMERIC\n')
    f.writelines('@ATTRIBUTE ratio_3_9 NUMERIC\n')
    f.writelines('@ATTRIBUTE ratio_22_9 NUMERIC\n')
    f.writelines('@ATTRIBUTE ratio_23_9 NUMERIC\n')
    f.writelines('@ATTRIBUTE ratio_24_9 NUMERIC\n')
    f.writelines('@ATTRIBUTE ratio_4_1 NUMERIC\n')
    f.writelines('@ATTRIBUTE ratio_5_1 NUMERIC\n')
    f.writelines('@ATTRIBUTE ratio_6_1 NUMERIC\n')     
    f.writelines('@ATTRIBUTE class {riaa_ok, riaa_ko}\n')
    f.writelines('@DATA\n')
    
    for file in os.listdir(inputDir):
        if file.endswith(".wav"):
            filename = os.path.join(inputDir, file)
            (absolute_bark, relative_bark, bark_ratios) = compute_bark_spectrum(filename)

            inputFile = filename

            if (inputFile.find('riaa_') != -1):
                riaa_class = 'riaa_ko'
            else:
                riaa_class = 'riaa_ok'
            
            ratios_array = numpy.asarray(bark_ratios)
            ratios = ["%.8f" % ratio for ratio in ratios_array]
            instance_tuple = (ratios, riaa_class)      
            instance_tuple = str(instance_tuple)
            instance_tuple.replace("[", "")
            instance_tuple.replace("]", "")
            instance_tuple.replace("'", "")
           
            f.writelines(instance_tuple)
            f.writelines ('\n')
    
    f.close
    
    return (bark_ratios, riaa_class)
    
    
def batch_bark_boxplot_computing(inputDir):

    BARK_BANDS = 24
    NUM_RATIOS = 9
    
    absolute_bark_array = []
    relative_bark_array = []
    bark_ratios_array = []

    for file in os.listdir(inputDir):
        if file.endswith(".wav"):
            filename = os.path.join(inputDir, file)
            (absolute_bark, relative_bark, bark_ratios) = compute_bark_spectrum(filename)
            absolute_bark_array.append(absolute_bark)
            relative_bark_array.append(relative_bark)
            bark_ratios_array.append(bark_ratios)

    absolute_bark_array = numpy.asarray(absolute_bark_array)
    relative_bark_array = numpy.asarray(relative_bark_array)
    bark_ratios_array = numpy.asarray(bark_ratios_array)

    boxplot_absolute_bark_matrix = []
    boxplot_relative_bark_matrix = []
    boxplot_bark_ratios_matrix = []

    for i in range(0,BARK_BANDS):
        boxplot_absolute_bark_matrix.append(absolute_bark_array[:,i])
        boxplot_relative_bark_matrix.append(relative_bark_array[:,i])
    
    plt.figure()
    plt.title('Absolute Bark')
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Bark band number')
    plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=0.7)
    plt.boxplot(boxplot_absolute_bark_matrix)
    axes = plt.gca()
    axes.yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.savefig(os.path.join(inputDir, 'absolute_bark.jpg'), dpi=600)
    
    plt.figure()
    plt.title('Relative Bark')
    plt.ylabel('Energy [%]')
    plt.xlabel('Bark band number')
    plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=0.7)
    plt.boxplot(boxplot_relative_bark_matrix)
    axes = plt.gca()
    axes.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    plt.savefig(os.path.join(inputDir, 'relative_bark.jpg'), dpi=600)
    
    for i in range(0,NUM_RATIOS):
         boxplot_bark_ratios_matrix.append(bark_ratios_array[:,i])
    
    plt.figure()
    plt.title('Bark ratios')
    plt.ylabel('Ratio value')
    plt.xlabel('Bark band number')
    plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=0.7)
    plt.rcParams['font.size'] = 5
    plt.boxplot(boxplot_bark_ratios_matrix, showfliers=False)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ['ratio_1_9', 'ratio_2_9', 
               'ratio_3_9', 'ratio_22_9', 'ratio_23_9', 'ratio_24_9', 
               'ratio_4_1', 'ratio_5_1', 'ratio_6_1'], rotation=45)
    axes = plt.gca()
    axes.yaxis.set_major_locator(ticker.MultipleLocator(0.10)) #MultipleLocator(0.05))
    plt.savefig(os.path.join(inputDir, 'bark_ratios.jpg'), dpi=600)


if __name__=='__main__':
    riaa_checker(sys.argv[1])