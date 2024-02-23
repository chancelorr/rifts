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
import geopandas as gpd
from sliderule import sliderule, icesat2, earthdata, h5, ipysliderule, io
from IPython import display
from datetime import datetime

#init sliderule
icesat2.init("slideruleearth.io", verbose=False)

#Some global vars
gtDict = {"gt1l": 10, "gt1r": 20, "gt2l": 30, "gt2r": 40, "gt3l": 50, "gt3r": 60, 
          10: "gt1l", 20: "gt1r", 30: "gt2l", 40: "gt2r", 50: "gt3l", 60: "gt3r"}

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

        
##############################################################################
################### Functions ################################################
##############################################################################

def display03Info(gdf):
    
    '''
    Display a bunch of info about the atl03 data:
    number of ground tracks, elevations
    available rgts, beams, cycles
    date range
    vertical range
    '''
    
    print("Reference Ground Tracks: {}".format(gdf["rgt"].unique()))
    #print("Beams: {}".format([gtDict[b] for b in list(atl06_sr['gt'].unique())]))
    print("Spots: {}".format(gdf["spot"].unique()))
    print("Cycles: {}".format(gdf["cycle"].unique()))
    print("Received {} elevations".format(gdf.shape[0]))
    print(f"Across {3*(len(np.unique(gdf.loc[:, 'rgt'])) + len(np.unique(gdf.loc[:, 'cycle'])))} strong tracks")
    # Exception here is only necessary when importing csv's
    try: print(f"Date range {gdf.index.min().date()} to {gdf.index.max().date()}")
    except: print(f"Date range {datetime.strptime(gdf.index.min()[:-3], '%Y-%m-%d %H:%M:%S.%f').date()}" 
        f" to {datetime.strptime(gdf.index.max()[:-3], '%Y-%m-%d %H:%M:%S.%f').date()}")
    return

#There is an error reading 
def display06Info(gdf):
    
    '''
    Display a bunch of info about the atl06 data:
    number of ground tracks, elevations
    available rgts, beams, cycles
    date range
    vertical range
    '''
    
    print("Reference Ground Tracks: {}".format(gdf["rgt"].unique()))
    print("Beams: {}".format([gtDict[b] for b in list(gdf['gt'].unique())]))
    print("Cycles: {}".format(gdf["cycle"].unique()))
    print("Received {} elevations".format(gdf.shape[0]))
    print(f"Across {3*(len(np.unique(gdf.loc[:, 'rgt'])) + len(np.unique(gdf.loc[:, 'cycle'])))} strong tracks")
    # Exception here is only necessary when importing csv's
    try: print(f"Date range {gdf.index.min().date()} to {gdf.index.max().date()}")
    except: print(f"Date range {pd.to_datetime(gdf.index, format='ISO8601').min().date()} to {pd.to_datetime(gdf.index, format='ISO8601').max().date()}")
    '''
    except: print(f"Date range {datetime.strptime(gdf.index.min()[:-3], '%Y-%m-%d %H:%M:%S.%f').date()}" 
        f" to {datetime.strptime(gdf.index.max()[:-3], '%Y-%m-%d %H:%M:%S.%f').date()}")
    '''
    print(f"Vertical range {gdf.h_mean.min()}m to {gdf.h_mean.max()}m")
    return

def filter_yapc(gdf, score=0):
    
    '''
    takes a gdf with yapc photons
    returns those with minimum score specified
    '''
    
    return gdf[gdf.weight_ph>score]
        
def get06Data(parms, file06_load=None, accessType=0, file06_save=None, sFlag06=0, verbose=False):
    
    '''
    Get ATL06 data using specified method (open .geojson, .csv, or process new using sliderule)
    If processing anew, save as .geojson or .csv if specified
    '''
    if not os.path.isfile(file06_load): accessType=0
    
    if accessType == 0:
        print('Processing new ATL06-SR dataset')

        # Request ATL06 Data
        atl06_sr = icesat2.atl06p(parms)
        atl06_sr = atl06_sr[(atl06_sr.spot==2)+(atl06_sr.spot==4)+(atl06_sr.spot==6)]

        # Display Statistics 
        if verbose: 
            try: display06Info(atl06_sr)
            except: print('Error displaying all info')

        # Save all data
        if sFlag06==1:
            # save geodataframe as geojson
            print(f'Saving file as geojson named {file06_save}.geojson')
            atl06_sr.to_file(f"{file06_save}.geojson", driver='GeoJSON')
        elif sFlag06==2:
            # Save geodataframe as csv
            print(f'Saving file as csv named {file06_save}.csv')
            atl06_sr.to_csv(f"{file06_save}.csv")

    elif accessType==1:
        #load from geojson
        print('Downloading atl06-SR data upload from .geojson file')
        atl06_sr = gpd.read_file(f"{file06_load}.geojson").set_index('time')
        if verbose: 
            try: display06Info(atl06_sr)
            except: print('Error displaying all info')
    elif accessType==2:
        # Load from csv (everything is a string)
        print('Downloading atl06-SR data upload from .csv file')
        atl06_sr = gpd.read_file(f"{file06_load}.csv").set_index('time')
        if verbose: display06Info(atl06_sr)
    elif accessType==-1:
        print('skipping data download altogether')
    return atl06_sr

def getDateTime(timestp):
    
    '''
    Get a datetime formatted for earthdata cmr search from geodataframe timestamp
    '''
    
    dat, tim = timestp.date(), timestp.time().strftime("%H:%M:%S")
    return f"{dat}T{tim}Z"

def getRegion(site, cycle):
    
    '''
    given site, cycle, opens available .geojson geometry and imports it as a geodataframe geometry
    '''
    
    try:     
        shpPath = f"../shapes/{shelf}/{site}_{cycle}.geojson"
        if not os.path.isfile(shpPath): shpPath = f"../shapes/{shelf}/{site}_00.geojson"
        # Read in EPSG:3031 shapefile and convert to EPSG:4326
        shpFile = gpd.read_file(shpPath)
        shpFile.crs = 'EPSG:3031'
        target_epsg = 'EPSG:4326'
        shpDF = shpFile.to_crs(target_epsg)
        region = sliderule.toregion(shpDF)["poly"]
    except: 
        print(f'Error getting region: .geojson geometry may not exist for {site}, cycle {cycle}')
        print(f' at path {shpPath}')
    return region

def getTrack(dat, trackInfo):
    
    '''
    Pull specific track (cycle, rgt, gt (beam) from geodataframe
    '''
    
    cycle, rgt, gt = trackInfo[0], trackInfo[1], gtDict[trackInfo[2]]
    track = dat[(dat.rgt==rgt)*(dat.cycle==cycle)*(dat.loc[:, 'gt']==gt)]
    return track, cycle, rgt, gt

def getTrackList(gdf, return_lens=False, omit=None, min_photons=200, verbose=False):
    
    '''
    Generate a list of available grount tracks in format cycle, rgt, gt (beam)
    '''
    
    rgtAll = list(gdf["rgt"].unique())
    gtAll = list(gdf['gt'].unique())
    cycleAll = list(gdf["cycle"].unique())
    lens = []
    tracks=[]
    
    if verbose:
        print(f'Finding tracks with minimum of {min_photons} photons')
        
    if type(omit)==list: 
        for k in omit: 
            rgtAll.remove(k)
        if verbose: print(f'Removed {len(omit)} tracks')
        
    for cyc in cycleAll:
        for r in rgtAll:
            for g in gtAll:
                tr = gdf[(gdf.cycle==cyc)*(gdf.rgt==r)*(gdf.loc[:, 'gt']==g)]
                if len(tr)>min_photons:
                    tracks.append((cyc, r, gtDict[g]))
                    lens.append(len(tr))
    
        
    if verbose: 
        print(f'{len(lens)} found')
        print(lens) 
    
    if return_lens: return tracks, lens
    return tracks

def makeIS2Map(data, interval=40):
    # Plot every n points from tracklist to not slow down the notebook
    projection = 'South'
    m = ipysliderule.leaflet(projection, zoom=10, scroll_wheel_zoom=True)
    m.GeoData(data.iloc[::40], column_name='h_mean', cmap='viridis')
    display.display(m.map)

def toGeojson(dat, filename):
    
    '''
    Save geodataframe as a .geojson
    '''
    
    #datReduced = dat.loc[:, ['cycle', 'rgt', 'spot', 'h_mean', 'geometry']]
    dat.to_file(f"{filename}.geojson", driver='GeoJSON')
    return

def unpackGranuleID(gid):
    '''
    Unpacks the granule ID specified by NSIDC:
    
    ATL[xx]_[yyyymmdd][hhmmss]_[tttt][cc][nn]_[vvv]_[rr].h5
    xx : ATLAS product number
    yyyymmdd : year, month and day of data acquisition
    hhmmss : start time, hour, minute, and second of data acquisition
    tttt : Reference Ground Track (RGT, ranges from 1–1387)
    cc : Orbital Cycle (91-day period)
    nn : Granule number (ranges from 1–14)
    vvv : data version number
    rr : data release number
    '''
    
    #global shortName, dat, tim, rgt, cycle, granuleNumber, version, release
    granID = gid[:]
    gid = gid.split('_')
    shortName = f'{gid[0]}'
    dat = f'{gid[1][0:4]}-{gid[1][4:6]}-{gid[1][6:8]}'
    tim = f'{gid[1][8:10]}:{gid[1][10:12]}:{gid[1][12:14]}'
    rgt = int(f'{gid[2][0:4]}')
    cycle = int(f'{gid[2][4:6]}')
    granuleNumber = int(f'{gid[2][6:8]}')
    version = int(f'{gid[3]}')
    release = int(f'{gid[4][0:2]}')
    return {"shortName": shortName, "date": dat, "time": tim, 
            "rgt": rgt, "cycle": cycle, "granuleNumber": granuleNumber, 
            "version": version, "release": release, "granuleID": granID}
    

######################################################################
####################### Unfinished/unused ############################
######################################################################
        

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
