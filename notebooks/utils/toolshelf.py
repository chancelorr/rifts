import numpy as np
import requests
import zipfile
import io
import os
import pprint
import re
import time
import h5py as h
import matplotlib.pyplot as plt

class granule:
    def __init__(self, hFile, product_short_name, beams='all'):
        f=h.File(hFile, 'r+')
        self.product = product_short_name
        
        #product dependent paths
        if self.product=='ATL06': 
            path='/land_ice_segments/'
            alt=path+'h_li'
            lon=path+'longitude'
            lat=path+'latitude'
        elif self.product=='ATL03': 
            path='/geolocation/reference_photon_'
            alt='heights'
            lon=path+'lon'
            lat=path+'lat'
        
        #all tracks
        all_tracks = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
        sc_orient = f['/orbit_info/sc_orient'][0]
        
        #0 is backward, wherein right beam is weak (0)
        #1 is forward, wherein right beam is strong (1) 
        #2 is transition, where all beams are (2)
        if sc_orient==0: 
            beam_strengths = [1, 0, 1, 0, 1, 0]
            strongs = [0, 2, 4]
            weaks = [1, 3, 5]
        elif sc_orient==1: 
            beam_strengths = [0, 1, 0, 1, 0, 1]
            strongs = [1, 3, 5]
            weaks = [0, 2, 4]
        elif sc_orient==2: 
            print('WARNING: sc_orient = \'transition\'. beam selection set to default \'all\'')
            beam_strengths = [2, 2, 2, 2, 2, 2]
            beams='all'
            
        #filter selected tracks
        if beams=='strong':
            tracks = [all_tracks[s] for s in strongs]
            beam_strengths = [beam_strengths[s] for s in strongs]
        elif beams=='weak':
            tracks = [all_tracks[w] for w in weaks]
            beam_strengths = [beam_strengths[w] for w in weaks]
        elif beams=='all':
            tracks = all_tracks
        elif beams!='all':
            print('ERROR: Invalid beam choice. Valid options are \'strong\', \'weak\', \'all\' (default)')
        
        #orbit stuff
        self.tracks = tracks
        self.beam_strength = beam_strengths
        self.strongs = strongs
        self.weaks = weaks
        self.sc_orient = sc_orient
        
        #data
        alts = [np.array(np.array(f[f'{t}{path}h_li']).tolist()) for t in tracks]
        lons = [f[f'{t}{path}longitude'][:] for t in tracks]
        lats = [f[f'{t}{path}latitude'][:] for t in tracks]
        self.alt = alts
        for a in alts:
            a[a>1e20] = float('nan')
        self.lon = lons
        self.lat = lats
        

# not in use currently, separates path directories and puts them into a list
def pullDirs(path):
	dirs = []
	path = path[1:]
	while len(path)>0:
		slash = path.find('/')
		dirs.append(path[:slash])
		path = path[(slash+1):]
	return dirs

def getPolygon(file):
    res = np.genfromtxt(file, delimiter=',')
    return res.reshape([int(len(res)/2), 2])

def getGranulePaths(shelfname):
    paths = []
    
def plotElevations(granule, tracks=[]):
    plt.figure()
