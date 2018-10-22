#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import errno
import logging
from datetime import datetime, timedelta
import re
import shutil
import sys


class StreamToLogger(object):
    """
    Redirect all the stdout/err to the logger, therefore both print and traceback
    are redirected to logger
    """

    def __init__(self, logger, LogFile='log', log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        self.logFile = LogFile

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=self.logFile,
            filemode='a'
        )
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def initLogger(logger_name, logfolder):
    """
    Init the logger with a name. Once initialized, all the module can use this logger simply getting a logger with the same name
    """
    logfilename = 'process_' + logger_name + '.log'
    logpath = os.path.join(logfolder, logfilename)
    if os.path.isfile(logpath):
        os.rename(logpath, logpath + '_old_' + str(os.path.getmtime(logpath)))  # so the name is always different
    else: #maybe the folder doesn't exist
        makedir(logfolder)

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)-18s %(levelname)-8s: %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logpath,
                        filemode='w')
    root = logging.getLogger('')  # https://stackoverflow.com/questions/15727420/using-python-logging-in-multiple-modules

    # define a Handler which writes DEBUG messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(filename)-18s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # ensure the flush at each log
    console.flush = sys.stdout.flush
    # add the handler to the root logger
    root.addHandler(console)

    #redirect stdout needed for logging message print from other library such keras
    sl = StreamToLogger(root, logpath, logging.INFO)
    sys.stdout = sl  # ovverride funcion

    #redirect stderr needed for logging execution error
    sl = StreamToLogger(root, logpath, logging.ERROR)
    sys.stderr = sl  # ovverride funcion

    logging.info('Logger initialized. ID='+logger_name)

def logcleaner():
    """
    Clean a text file
    :param pathFile:
    :return:
    """
    nameFileLog = logging.getLoggerClass().root.handlers[0].baseFilename
    if os.path.isfile(nameFileLog):

        string = open(nameFileLog).read()
        new_str = re.sub('[^a-zA-Z0-9\n\.\-=<>~:,\[\]\t_(){}/]', ' ', string)
        with open(nameFileLog, 'w') as logfile:
            logfile.write(new_str)

    return


def deleteContentFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def GetTime(s):
    '''
    :param s: seconds (int)
    :return: the days hours minutes and second that corresponds to the seconds input, as string
    '''
    d = datetime(1, 1, 1) + s
    t = "%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second)
    return t


def path_creation(str_ID, exp_ID=0, root_path=None):

    # TODO split string_ID in exp_ID e str_ID
    if root_path is None:
        root_path = os.path.realpath("..")
    experiments_base_path = os.path.join(root_path, 'experiments')
    makedir(experiments_base_path)
    experiment_folder = os.path.join(experiments_base_path , str_ID)
    # check if experiment folder altready exists
    count = 0
    if not exp_ID and os.path.isdir(experiment_folder):
        for dirs in os.listdir(experiments_base_path):
            if str_ID in dirs and os.path.isdir(os.path.join(experiments_base_path, dirs)):
                count += 1
        experiment_folder = experiment_folder + '.' + str(count)

    return root_path, experiment_folder


# utility function
def makedir(path):
    """
    Make dir only is it doesn't exist yet
    :param path: path to the folder that is to be created
    :return:
    """
    try:
        os.makedirs(path)
        #logging.info("Make " + path + " dir")
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        pass
    return


def item_to_dictionary(item,keys):
    """converts item(row) of Dataframe into dictionary, given a list of keys

    Parameters
    ----------
    item: DataFrame row - Series
    keys: a list containing dictionary keys = columns of Dataframe

    Returns
    -------
    item converted into dictionary

    """
    dict = {}
    for k in keys:
        value = item[k]
        dict.update({k: value})

    return dict
