import numpy as np
import csv
#import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', action='store', dest='nstep', type=int, default=1)
parser.add_argument('--nodev', action='store_true', dest='noDev', default=False)
argu = parser.parse_args()

nstep = argu.nstep
noDev = argu.noDev

##
####1111
