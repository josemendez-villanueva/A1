# For testing purposes will comnine netParams and cfg

"""
netParams.py 

High-level specifications for A1 network model using NetPyNE

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""




# try:
# 	from __main__ import cfg  # import SimConfig object with params from parent module
# except:
# 	from cfg import cfg



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CFG 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SIMULATION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#cfg aspect of the simulation

"""
cfg.py 

Simulation configuration for A1 model (using NetPyNE)
This file has sim configs as well as specification for parameterized values in netParams.py 

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""



        
from lib2to3.pgen2.pgen import DFAState
from netpyne import specs, sim, analysis
import pickle, json


from netpyne import specs
import numpy as np
import pickle



netParams = specs.NetParams()   # object of class NetParams to store the network parameters

cfg = specs.SimConfig()



#~~~~~~~~~~~~~~~~~~~~~~~    
#Set the param toa default as a function parameter


def main(x = [1,1,1,1,1,1,1,1,1,1,1,1,1]):


    #Clean the parameter list up, it is in this format due to different testing/applications
    #Parameter list that is passed in
    recent_param = x

    #parameters that are imported
    paramip = [recent_param[0], recent_param[1],recent_param[2],recent_param[3],recent_param[4],recent_param[5],recent_param[6],recent_param[7],recent_param[8],recent_param[9],recent_param[10],recent_param[11],recent_param[12]]

    #To see if each parallel sim is actually receiving the different parameter
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(paramip)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    

    #~~~~~~~~~~~~~~~~~~~~~~~ 

    #------------------------------------------------------------------------------
    #
    # SIMULATION CONFIGURATION
    #
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Run parameters
    #------------------------------------------------------------------------------
    cfg.duration = 1500 #1.5*1e3			## Duration of the sim, in ms -- value from M1 cfg.py 
    cfg.dt = 0.05                   ## Internal Integration Time Step -- value from M1 cfg.py 
    cfg.verbose = 0         	## Show detailed messages
    cfg.hParams['celsius'] = 37
    cfg.createNEURONObj = 1
    cfg.createPyStruct = 1

    cfg.connRandomSecFromList = False  # set to false for reproducibility 
    cfg.cvode_active = False
    cfg.cvode_atol = 1e-6
    cfg.cache_efficient = True
    cfg.printRunTime = 0.1
    cfg.oneSynPerNetcon = False
    cfg.includeParamsLabel = False
    cfg.printPopAvgRates = [500, cfg.duration]

    #------------------------------------------------------------------------------
    # Recording 
    #------------------------------------------------------------------------------
    cfg.allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC']

    alltypes = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'ITS4', 'PT5B', 'TC', 'HTC', 'IRE', 'TI']

    cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc': 0.5, 'var':'v'}}  ## Dict with traces to record -- taken from M1 cfg.py 
    cfg.recordStim = True			## Seen in M1 cfg.py
    cfg.recordTime = True  		## SEen in M1 cfg.py 
    cfg.recordStep = 0.1            ## Step size (in ms) to save data -- value from M1 cfg.py 

    cfg.recordLFP = [[100, y, 100] for y in range(0, 2000, 100)] #+[[100, 2500, 200], [100,2700,200]]
    #cfg.recordLFP = [[x, 1000, 100] for x in range(100, 2200, 200)] #+[[100, 2500, 200], [100,2700,200]]
    #cfg.saveLFPPops =  ["IT3"] #, "IT3", "SOM3", "PV3", "VIP3", "NGF3", "ITP4", "ITS4", "IT5A", "CT5A", "IT5B", "PT5B", "CT5B", "IT6", "CT6"]

    # cfg.recordDipole = True
    # cfg.saveDipoleCells = ['all']
    # cfg.saveDipolePops = cfg.allpops


    #------------------------------------------------------------------------------
    # Saving
    #------------------------------------------------------------------------------

    cfg.simLabel = 'v31_tune3' 
    cfg.saveFolder = 'data/v31_manualTune'                	## Set file output name
    cfg.savePickle = False      	## Save pkl file
    cfg.saveJson = False         	## Save json file
    cfg.timestampFilename = True
    cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net'] 
    cfg.backupCfgFile = None 		
    cfg.gatherOnlySimData = False	 
    cfg.saveCellSecs = True
    cfg.saveCellConns = True		 

    #------------------------------------------------------------------------------
    # Analysis and plotting 
    #----------------------------------------------------------------------------- 
    #

    cfg.analysis['plotTraces'] = {'include': [(pop, 0) for pop in cfg.allpops], 'oneFigPer': 'trace', 'overlay': True, 'saveFig': True, 'showFig': True, 'figSize':(12,8)} #[(pop,0) for pop in alltypes]		## Seen in M1 cfg.py (line 68) 
    cfg.analysis['plotRaster'] = {'include': cfg.allpops, 'saveFig': True, 'showFig': False, 'popRates': True, 'orderInverse': True, 'timeRange': [0,cfg.duration], 'figSize': (14,12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300}      	## Plot a raster
    #cfg.analysis['plotSpikeStats'] = {'stats': ['rate'], 'figSize': (6,12), 'timeRange': [0, 2500], 'dpi': 300, 'showFig': 0, 'saveFig': 1}

    cfg.analysis['plotLFP'] = {'plots': ['timeSeries'], 'electrodes': [10], 'maxFreq': 80, 'figSize': (8,4), 'saveData': False, 'saveFig': True, 'showFig': False} # 'PSD', 'spectrogram'
    #cfg.analysis['plotDipole'] = {'saveFig': True}
    #cfg.analysis['plotEEG'] = {'saveFig': True}


    #layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}
    #cfg.analysis['plotCSD'] = {'spacing_um': 100, 'LFP_overlay': 1, 'layer_lines': 1, 'layer_bounds': layer_bounds, 'saveFig': 1, 'showFig': 0}
    #cfg.analysis['plot2Dnet'] = True      	## Plot 2D visualization of cell positions & connections 


    #------------------------------------------------------------------------------
    # Cells
    #------------------------------------------------------------------------------
    cfg.weightNormThreshold = 5.0  # maximum weight normalization factor with respect to the soma
    cfg.weightNormScaling = {'NGF_reduced': 1.0, 'ITS4_reduced': 1.0}


    #------------------------------------------------------------------------------
    # Synapses
    #------------------------------------------------------------------------------
    cfg.AMPATau2Factor = 1.0
    cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
    cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
    cfg.synWeightFractionSOME = [0.9, 0.1] # SOM -> E GABAASlow to GABAB ratio
    cfg.synWeightFractionNGF = [0.5, 0.5] # NGF GABAA to GABAB ratio
    cfg.synWeightFractionENGF = [0.834, 0.166] # NGF AMPA to NMDA ratio


    #------------------------------------------------------------------------------
    # Network 
    #------------------------------------------------------------------------------
    ## These values taken from M1 cfg.py (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/cfg.py)
    cfg.singleCellPops = False
    cfg.singlePop = ''
    cfg.removeWeightNorm = False
    cfg.scale = 1.0     # 1.0     # Is this what should be used? 
    cfg.sizeY = 2000.0 ##should be 2000 #1350.0 in M1_detailed # should this be set to 2000 since that is the full height of the column? 
    cfg.sizeX = 200.0 ##should be 200    # 400 - This may change depending on electrode radius 
    cfg.sizeZ = 200.0 ##should be 200
    cfg.scaleDensity = 0.075 #1.0 #0.075 # Should be 1.0 unless need lower cell density for test simulation or visualization





    #------------------------------------------------------------------------------
    # Connectivity
    #------------------------------------------------------------------------------
    cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
    cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
    cfg.synWeightFractionIE = [0.9, 0.1]  # SOM -> E GABAASlow to GABAB ratio (update this)
    cfg.synWeightFractionII = [0.9, 0.1]  # SOM -> E GABAASlow to GABAB ratio (update this)

    # Cortical
    cfg.addConn = 1



    #cfg.EEGain = 1.0
    #cfg.EIGain = 1.0 # 1.8600534795309025 	
    cfg.IEGain = 1.0 #0.75
    cfg.IIGain = 1.0 #0.5



    ## E/I->E/I layer weights (L1-3, L4, L5, L6)
    cfg.EELayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}
    cfg.EILayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}

    #cfg.IELayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}
    #cfg.IILayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}



    #################### Paramip to params needed to be changed

    ####################

    cfg.EEGain = paramip[0]
    cfg.EIGain = paramip[1] # 1.8600534795309025 	


    cfg.IELayerGain = {'1': paramip[2], '2': paramip[2], '3': paramip[2], '4': paramip[3] , '5A': paramip[4], '5B': paramip[4], '6': paramip[5]}
    cfg.IILayerGain = {'1': paramip[6], '2': paramip[6], '3': paramip[6], '4': paramip[7] , '5A': paramip[8], '5B': paramip[8], '6': paramip[9]}


    cfg.thalamoCorticalGain = paramip[10] 
    cfg.intraThalamicGain = paramip[11]
    cfg.corticoThalamicGain = paramip[12]

    ####################
    ####################

    ## E->I by target cell type
    cfg.EICellTypeGain= {'PV': 1.0, 'SOM': 1.0, 'VIP': 1.0, 'NGF': 1.0}

    ## I->E by target cell type
    cfg.IECellTypeGain= {'PV': 1.0, 'SOM': 1.0, 'VIP': 1.0, 'NGF': 1.0}

    # Thalamic
    cfg.addIntraThalamicConn = 1.0
    cfg.addIntraThalamicConn = 1.0
    cfg.addCorticoThalamicConn = 1.0
    cfg.addThalamoCorticalConn = 1.0

    #cfg.thalamoCorticalGain = 1.0 
    #cfg.intraThalamicGain = 1.0
    #cfg.corticoThalamicGain = 1.0

    cfg.addSubConn = 1

    ## full weight conn matrix
    with open('conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
    cfg.wmat = connData['wmat']


    #------------------------------------------------------------------------------
    # Background inputs
    #------------------------------------------------------------------------------
    cfg.addBkgConn = 1
    cfg.noiseBkg = 1.0  # firing rate random noise
    cfg.delayBkg = 5.0  # (ms)
    cfg.startBkg = 0  # start at 0 ms

    # cfg.weightBkg = {'IT': 12.0, 'ITS4': 0.7, 'PT': 14.0, 'CT': 14.0,
    #                 'PV': 28.0, 'SOM': 5.0, 'NGF': 80.0, 'VIP': 9.0,
    #                 'TC': 1.8, 'HTC': 1.55, 'RE': 9.0, 'TI': 3.6}
    cfg.rateBkg = {'exc': 40, 'inh': 40}

    ## options to provide external sensory input
    #cfg.randomThalInput = True  # provide random bkg inputs spikes (NetStim) to thalamic populations 

    cfg.EbkgThalamicGain = 4.0
    cfg.IbkgThalamicGain = 4.0

    cfg.cochlearThalInput = False #{'numCells': 200, 'freqRange': [9*1e3, 11*1e3], 'toneFreq': 10*1e3, 'loudnessDBs': 50}  # parameters to generate realistic  auditory thalamic inputs using Brian Hears 

    # parameters to generate realistic cochlear + IC input ; weight =unitary connection somatic EPSP (mV)
    cfg.ICThalInput = {} #'file': 'data/ICoutput/ICoutput_CF_9600_10400_wav_01_ba_peter.mat', 
                        #'startTime': 500, 'weightE': 0.5, 'weightI': 0.5, 'probE': 0.12, 'probI': 0.26, 'seed': 1}  

    #------------------------------------------------------------------------------
    # Current inputs 
    #------------------------------------------------------------------------------
    cfg.addIClamp = 0

    #------------------------------------------------------------------------------
    # NetStim inputs 
    #------------------------------------------------------------------------------

    cfg.addNetStim = 0

    ## LAYER 1
    cfg.NetStim1 = {'pop': 'NGF1', 'ynorm': [0,2.0], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 0.0, 'weight': 10.0, 'delay': 0}

    # ## LAYER 2
    # cfg.NetStim2 = {'pop': 'IT2',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}


    cfg.tune = {}





    #------------------------------------------------------------------------------
    # VERSION 
    #------------------------------------------------------------------------------
    netParams.version = 34

    #------------------------------------------------------------------------------
    #
    # NETWORK PARAMETERS
    #
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # General network parameters
    #------------------------------------------------------------------------------

    netParams.scale = cfg.scale # Scale factor for number of cells # NOT DEFINED YET! 3/11/19 # How is this different than scaleDensity? 
    netParams.sizeX = cfg.sizeX # x-dimension (horizontal length) size in um
    netParams.sizeY = cfg.sizeY # y-dimension (vertical height or cortical depth) size in um
    netParams.sizeZ = cfg.sizeZ # z-dimension (horizontal depth) size in um
    netParams.shape = 'cylinder' # cylindrical (column-like) volume

    #------------------------------------------------------------------------------
    # General connectivity parameters
    #------------------------------------------------------------------------------
    netParams.scaleConnWeight = 1.0 # Connection weight scale factor (default if no model specified)
    netParams.scaleConnWeightModels = { 'HH_reduced': 1.0, 'HH_full': 1.0} #scale conn weight factor for each cell model
    netParams.scaleConnWeightNetStims = 1.0 #0.5  # scale conn weight factor for NetStims
    netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells
    netParams.defaultDelay = 2.0 # default conn delay (ms)
    netParams.propVelocity = 500.0 # propagation velocity (um/ms)
    netParams.probLambda = 100.0  # length constant (lambda) for connection probability decay (um)

    #------------------------------------------------------------------------------
    # Cell parameters
    #------------------------------------------------------------------------------

    Etypes = ['IT', 'ITS4', 'PT', 'CT']
    Itypes = ['PV', 'SOM', 'VIP', 'NGF']
    cellModels = ['HH_reduced', 'HH_full'] # List of cell models

    # II: 100-950, IV: 950-1250, V: 1250-1550, VI: 1550-2000 
    layer = {'1': [0.00, 0.05], '2': [0.05, 0.08], '3': [0.08, 0.475], '4': [0.475, 0.625], '5A': [0.625, 0.667], '5B': [0.667, 0.775], '6': [0.775, 1], 'thal': [1.2, 1.4], 'cochlear': [1.6, 1.8]}  # normalized layer boundaries  

    layerGroups = { '1-3': [layer['1'][0], layer['3'][1]],  # L1-3
                    '4': layer['4'],                      # L4
                    '5': [layer['5A'][0], layer['5B'][1]],  # L5A-5B
                    '6': layer['6']}                        # L6

    # add layer border correction ??
    #netParams.correctBorder = {'threshold': [cfg.correctBorderThreshold, cfg.correctBorderThreshold, cfg.correctBorderThreshold], 
    #                        'yborders': [layer['2'][0], layer['5A'][0], layer['6'][0], layer['6'][1]]}  # correct conn border effect


    #------------------------------------------------------------------------------
    ## Load cell rules previously saved using netpyne format (DOES NOT INCLUDE VIP, NGF and spiny stellate)
    ## include conditions ('conds') for each cellRule
    cellParamLabels = ['IT2_reduced', 'IT3_reduced', 'ITP4_reduced', 'ITS4_reduced',
                        'IT5A_reduced', 'CT5A_reduced', 'IT5B_reduced',
                        'PT5B_reduced', 'CT5B_reduced', 'IT6_reduced', 'CT6_reduced',
                        'PV_reduced', 'SOM_reduced', 'VIP_reduced', 'NGF_reduced',
                        'RE_reduced', 'TC_reduced', 'HTC_reduced', 'TI_reduced']

    for ruleLabel in cellParamLabels:
        netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/' + ruleLabel + '_cellParams.json')  # Load cellParams for each of the above cell subtype

    # change weightNorm 
    for k in cfg.weightNormScaling:
        for sec in netParams.cellParams[k]['secs'].values():
            for i in range(len(sec['weightNorm'])):
                sec['weightNorm'][i] *= cfg.weightNormScaling[k]



    #------------------------------------------------------------------------------
    # Population parameters
    #------------------------------------------------------------------------------

    ## load densities
    with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']
    density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

    # ### LAYER 1:
    netParams.popParams['NGF1'] = {'cellType': 'NGF', 'cellModel': 'HH_reduced','ynormRange': layer['1'],   'density': density[('A1','nonVIP')][0]}

    ### LAYER 2:
    netParams.popParams['IT2'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['2'],   'density': density[('A1','E')][1]}     # cfg.cellmod for 'cellModel' in M1 netParams.py 
    netParams.popParams['SOM2'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','SOM')][1]}   
    netParams.popParams['PV2'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','PV')][1]}    
    netParams.popParams['VIP2'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','VIP')][1]}
    netParams.popParams['NGF2'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','nonVIP')][1]}

    ### LAYER 3:
    netParams.popParams['IT3'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['3'],   'density': density[('A1','E')][1]} ## CHANGE DENSITY
    netParams.popParams['SOM3'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','SOM')][1]} ## CHANGE DENSITY
    netParams.popParams['PV3'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','PV')][1]} ## CHANGE DENSITY
    netParams.popParams['VIP3'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','VIP')][1]} ## CHANGE DENSITY
    netParams.popParams['NGF3'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','nonVIP')][1]}


    ### LAYER 4: 
    netParams.popParams['ITP4'] =	 {'cellType': 'IT', 'cellModel': 'HH_reduced',  'ynormRange': layer['4'],   'density': 0.5*density[('A1','E')][2]}      ## CHANGE DENSITY #
    netParams.popParams['ITS4'] =	 {'cellType': 'IT', 'cellModel': 'HH_reduced', 'ynormRange': layer['4'],  'density': 0.5*density[('A1','E')][2]}      ## CHANGE DENSITY 
    netParams.popParams['SOM4'] = 	 {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','SOM')][2]}
    netParams.popParams['PV4'] = 	 {'cellType': 'PV', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],   'density': density[('A1','PV')][2]}
    netParams.popParams['VIP4'] =	 {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','VIP')][2]}
    netParams.popParams['NGF4'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','nonVIP')][2]}

    # # ### LAYER 5A: 
    netParams.popParams['IT5A'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'], 	'density': 0.5*density[('A1','E')][3]}      
    netParams.popParams['CT5A'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'],   'density': 0.5*density[('A1','E')][3]}  # density is [5] because we are using same numbers for L5A and L6 for CT cells? 
    netParams.popParams['SOM5A'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','SOM')][3]}          
    netParams.popParams['PV5A'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','PV')][3]}         
    netParams.popParams['VIP5A'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','VIP')][3]}
    netParams.popParams['NGF5A'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','nonVIP')][3]}

    ### LAYER 5B: 
    netParams.popParams['IT5B'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': (1/3)*density[('A1','E')][4]}  
    netParams.popParams['CT5B'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'],   'density': (1/3)*density[('A1','E')][4]}  # density is [5] because we are using same numbers for L5B and L6 for CT cells? 
    netParams.popParams['PT5B'] =     {'cellType': 'PT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': (1/3)*density[('A1','E')][4]}  
    netParams.popParams['SOM5B'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1', 'SOM')][4]}
    netParams.popParams['PV5B'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],	'density': density[('A1','PV')][4]}     
    netParams.popParams['VIP5B'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1','VIP')][4]}
    netParams.popParams['NGF5B'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1','nonVIP')][4]}

    # # ### LAYER 6:
    netParams.popParams['IT6'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]}  
    netParams.popParams['CT6'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]} 
    netParams.popParams['SOM6'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','SOM')][5]}   
    netParams.popParams['PV6'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','PV')][5]}     
    netParams.popParams['VIP6'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','VIP')][5]}
    netParams.popParams['NGF6'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','nonVIP')][5]}


    ### THALAMIC POPULATIONS (from prev model)
    thalDensity = density[('A1','PV')][2] * 1.25  # temporary estimate (from prev model)

    netParams.popParams['TC'] =     {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.75*thalDensity}  
    netParams.popParams['TCM'] =    {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity} 
    netParams.popParams['HTC'] =    {'cellType': 'HTC', 'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.25*thalDensity}   
    netParams.popParams['IRE'] =    {'cellType': 'RE',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity}     
    netParams.popParams['IREM'] =   {'cellType': 'RE', 'cellModel': 'HH_reduced',   'ynormRange': layer['thal'],   'density': thalDensity}
    netParams.popParams['TI'] =     {'cellType': 'TI',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.33 * thalDensity} ## Winer & Larue 1996; Huang et al 1999 
    netParams.popParams['TIM'] =    {'cellType': 'TI',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.33 * thalDensity} ## Winer & Larue 1996; Huang et al 1999 


    if cfg.singleCellPops:
        for pop in netParams.popParams.values(): pop['numCells'] = 1

    ## List of E and I pops to use later on
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B' , 'PT5B', 'IT6', 'CT6']  # all layers

    Ipops = ['NGF1',                            # L1
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            'PV6', 'SOM6', 'VIP6', 'NGF6']      # L6 



    #------------------------------------------------------------------------------
    # Synaptic mechanism parameters
    #------------------------------------------------------------------------------

    ### From M1 detailed netParams.py 
    netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
    netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3*cfg.AMPATau2Factor, 'e': 0}
    netParams.synMechParams['GABAB'] = {'mod':'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
    netParams.synMechParams['GABAA'] = {'mod':'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
    netParams.synMechParams['GABAA_VIP'] = {'mod':'MyExp2SynBB', 'tau1': 0.3, 'tau2': 6.4, 'e': -80}  # Pi et al 2013
    netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB','tau1': 2, 'tau2': 100, 'e': -80}
    netParams.synMechParams['GABAASlowSlow'] = {'mod': 'MyExp2SynBB', 'tau1': 200, 'tau2': 400, 'e': -80}

    ESynMech = ['AMPA', 'NMDA']
    SOMESynMech = ['GABAASlow','GABAB']
    SOMISynMech = ['GABAASlow']
    PVSynMech = ['GABAA']
    VIPSynMech = ['GABAA_VIP']
    NGFSynMech = ['GABAA', 'GABAB']


    #------------------------------------------------------------------------------
    # Local connectivity parameters
    #------------------------------------------------------------------------------

    ## load data from conn pre-processing file
    with open('conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
    pmat = connData['pmat']
    lmat = connData['lmat']
    wmat = connData['wmat']
    bins = connData['bins']
    connDataSource = connData['connDataSource']

    wmat = cfg.wmat

    layerGainLabels = ['1', '2', '3', '4', '5A', '5B', '6']

    #------------------------------------------------------------------------------
    ## E -> E
    if cfg.addConn and cfg.EEGain > 0.0:
        for pre in Epops:
            for post in Epops:
                for l in layerGainLabels:  # used to tune each layer group independently
                    if connDataSource['E->E/I'] in ['Allen_V1', 'Allen_custom']:
                        prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                    else:
                        prob = pmat[pre][post]
                    netParams.connParams['EE_'+pre+'_'+post+'_'+l] = { 
                        'preConds': {'pop': pre}, 
                        'postConds': {'pop': post, 'ynorm': layer[l]},
                        'synMech': ESynMech,
                        'probability': prob,
                        'weight': wmat[pre][post] * cfg.EEGain * cfg.EELayerGain[l], 
                        'synMechWeightFactor': cfg.synWeightFractionEE,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': 1,
                        'sec': 'dend_all'}
                        

    #------------------------------------------------------------------------------
    ## E -> I
    if cfg.addConn and cfg.EIGain > 0.0:
        for pre in Epops:
            for post in Ipops:
                for postType in Itypes:
                    if postType in post: # only create rule if celltype matches pop
                        for l in layerGainLabels:  # used to tune each layer group independently
                            if connDataSource['E->E/I'] in ['Allen_V1', 'Allen_custom']:
                                prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                            else:
                                prob = pmat[pre][post]
                            
                            if 'NGF' in post:
                                synWeightFactor = cfg.synWeightFractionENGF
                            else:
                                synWeightFactor = cfg.synWeightFractionEI
                            netParams.connParams['EI_'+pre+'_'+post+'_'+postType+'_'+l] = { 
                                'preConds': {'pop': pre}, 
                                'postConds': {'pop': post, 'cellType': postType, 'ynorm': layer[l]},
                                'synMech': ESynMech,
                                'probability': prob,
                                'weight': wmat[pre][post] * cfg.EIGain * cfg.EICellTypeGain[postType] * cfg.EILayerGain[l], 
                                'synMechWeightFactor': synWeightFactor,
                                'delay': 'defaultDelay+dist_3D/propVelocity',
                                'synsPerConn': 1,
                                'sec': 'proximal'}
                    

    #------------------------------------------------------------------------------
    ## I -> E
    if cfg.addConn and cfg.IEGain > 0.0:

        if connDataSource['I->E/I'] == 'Allen_custom':

            ESynMech = ['AMPA', 'NMDA']
            SOMESynMech = ['GABAASlow','GABAB']
            SOMISynMech = ['GABAASlow']
            PVSynMech = ['GABAA']
            VIPSynMech = ['GABAA_VIP']
            NGFSynMech = ['GABAA', 'GABAB']

            for pre in Ipops:
                for preType in Itypes:
                    if preType in pre:  # only create rule if celltype matches pop
                        for post in Epops:
                            for l in layerGainLabels:  # used to tune each layer group independently
                                
                                prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                                
                                if 'SOM' in pre:
                                    synMech = SOMESynMech
                                elif 'PV' in pre:
                                    synMech = PVSynMech
                                elif 'VIP' in pre:
                                    synMech = VIPSynMech
                                elif 'NGF' in pre:
                                    synMech = NGFSynMech

                                netParams.connParams['IE_'+pre+'_'+preType+'_'+post+'_'+l] = { 
                                    'preConds': {'pop': pre}, 
                                    'postConds': {'pop': post, 'ynorm': layer[l]},
                                    'synMech': synMech,
                                    'probability': prob,
                                    'weight': wmat[pre][post] * cfg.IEGain * cfg.IECellTypeGain[preType] * cfg.IELayerGain[l], 
                                    'synMechWeightFactor': cfg.synWeightFractionEI,
                                    'delay': 'defaultDelay+dist_3D/propVelocity',
                                    'synsPerConn': 1,
                                    'sec': 'proximal'}
                        

    #------------------------------------------------------------------------------
    ## I -> I
    if cfg.addConn and cfg.IIGain > 0.0:

        if connDataSource['I->E/I'] == 'Allen_custom':

            for pre in Ipops:
                for post in Ipops:
                    for l in layerGainLabels: 
                        
                        prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])

                        if 'SOM' in pre:
                            synMech = SOMISynMech
                        elif 'PV' in pre:
                            synMech = PVSynMech
                        elif 'VIP' in pre:
                            synMech = VIPSynMech
                        elif 'NGF' in pre:
                            synMech = NGFSynMech

                        netParams.connParams['II_'+pre+'_'+post+'_'+l] = { 
                            'preConds': {'pop': pre}, 
                            'postConds': {'pop': post,  'ynorm': layer[l]},
                            'synMech': synMech,
                            'probability': prob,
                            'weight': wmat[pre][post] * cfg.IIGain * cfg.IILayerGain[l], 
                            'synMechWeightFactor': cfg.synWeightFractionII,
                            'delay': 'defaultDelay+dist_3D/propVelocity',
                            'synsPerConn': 1,
                            'sec': 'proximal'}
                            

    #------------------------------------------------------------------------------
    # Thalamic connectivity parameters
    #------------------------------------------------------------------------------


    #------------------------------------------------------------------------------
    ## Intrathalamic 

    TEpops = ['TC', 'TCM', 'HTC']
    TIpops = ['IRE', 'IREM', 'TI', 'TIM']

    if cfg.addConn and cfg.addIntraThalamicConn:
        for pre in TEpops+TIpops:
            for post in TEpops+TIpops:
                if post in pmat[pre]:
                    # for syns use ESynMech, SOMESynMech and SOMISynMech 
                    if pre in TEpops:     # E->E/I
                        syn = ESynMech
                        synWeightFactor = cfg.synWeightFractionEE
                    elif post in TEpops:  # I->E
                        syn = SOMESynMech
                        synWeightFactor = cfg.synWeightFractionIE
                    else:                  # I->I
                        syn = SOMISynMech
                        synWeightFactor = [1.0]
                        
                    netParams.connParams['ITh_'+pre+'_'+post] = { 
                        'preConds': {'pop': pre}, 
                        'postConds': {'pop': post},
                        'synMech': syn,
                        'probability': pmat[pre][post],
                        'weight': wmat[pre][post] * cfg.intraThalamicGain, 
                        'synMechWeightFactor': synWeightFactor,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': 1,
                        'sec': 'soma'}  


    #------------------------------------------------------------------------------
    ## Corticothalamic 
    if cfg.addConn and cfg.addCorticoThalamicConn:
        for pre in Epops:
            for post in TEpops+TIpops:
                if post in pmat[pre]:
                    netParams.connParams['CxTh_'+pre+'_'+post] = { 
                        'preConds': {'pop': pre}, 
                        'postConds': {'pop': post},
                        'synMech': ESynMech,
                        'probability': pmat[pre][post],
                        'weight': wmat[pre][post] * cfg.corticoThalamicGain, 
                        'synMechWeightFactor': cfg.synWeightFractionEE,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': 1,
                        'sec': 'soma'}  

    #------------------------------------------------------------------------------
    ## Thalamocortical 
    if cfg.addConn and cfg.addThalamoCorticalConn:
        for pre in TEpops+TIpops:
            for post in Epops+Ipops:
                if post in pmat[pre]:
                    # for syns use ESynMech, SOMESynMech and SOMISynMech 
                    if pre in TEpops:     # E->E/I
                        syn = ESynMech
                        synWeightFactor = cfg.synWeightFractionEE
                    elif post in Epops:  # I->E
                        syn = SOMESynMech
                        synWeightFactor = cfg.synWeightFractionIE
                    else:                  # I->I
                        syn = SOMISynMech
                        synWeightFactor = [1.0]

                    netParams.connParams['ThCx_'+pre+'_'+post] = { 
                        'preConds': {'pop': pre}, 
                        'postConds': {'pop': post},
                        'synMech': syn,
                        'probability': '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post]),
                        'weight': wmat[pre][post] * cfg.thalamoCorticalGain, 
                        'synMechWeightFactor': synWeightFactor,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': 1,
                        'sec': 'soma'}  


    #------------------------------------------------------------------------------
    # Subcellular connectivity (synaptic distributions)
    #------------------------------------------------------------------------------  

    # Set target sections (somatodendritic distribution of synapses)
    # From Billeh 2019 (Allen V1) (fig 4F) and Tremblay 2016 (fig 3)

    if cfg.addSubConn:
        #------------------------------------------------------------------------------
        # E -> E2/3,4: soma,dendrites <200um
        netParams.subConnParams['E->E2,3,4'] = {
            'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
            'postConds': {'pops': ['IT2', 'IT3', 'ITP4', 'ITS4']},
            'sec': 'proximal',
            'groupSynMechs': ESynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        # E -> E5,6: soma,dendrites (all)
        netParams.subConnParams['E->E5,6'] = {
            'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
            'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
            'sec': 'all',
            'groupSynMechs': ESynMech, 
            'density': 'uniform'}
            
        #------------------------------------------------------------------------------
        # E -> I: soma, dendrite (all)
        netParams.subConnParams['E->I'] = {
            'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
            'postConds': {'cellType': ['PV','SOM','NGF', 'VIP']},
            'sec': 'all',
            'groupSynMechs': ESynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        # NGF1 -> E: apic_tuft
        netParams.subConnParams['NGF1->E'] = {
            'preConds': {'pops': ['NGF1']}, 
            'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
            'sec': 'apic_tuft',
            'groupSynMechs': NGFSynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        # NGF2,3,4 -> E2,3,4: apic_trunk
        netParams.subConnParams['NGF2,3,4->E2,3,4'] = {
            'preConds': {'pops': ['NGF2', 'NGF3', 'NGF4']}, 
            'postConds': {'pops': ['IT2', 'IT3', 'ITP4', 'ITS4']},
            'sec': 'apic_trunk',
            'groupSynMechs': NGFSynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        # NGF2,3,4 -> E5,6: apic_uppertrunk
        netParams.subConnParams['NGF2,3,4->E5,6'] = {
            'preConds': {'pops': ['NGF2', 'NGF3', 'NGF4']}, 
            'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
            'sec': 'apic_uppertrunk',
            'groupSynMechs': NGFSynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        # NGF5,6 -> E5,6: apic_lowerrunk
        netParams.subConnParams['NGF5,6->E5,6'] = {
            'preConds': {'pops': ['NGF5A', 'NGF5B', 'NGF6']}, 
            'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
            'sec': 'apic_lowertrunk',
            'groupSynMechs': NGFSynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        #  SOM -> E: all_dend (not close to soma)
        netParams.subConnParams['SOM->E'] = {
            'preConds': {'cellType': ['SOM']}, 
            'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
            'sec': 'dend_all',
            'groupSynMechs': SOMESynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        #  PV -> E: proximal
        netParams.subConnParams['PV->E'] = {
            'preConds': {'cellType': ['PV']}, 
            'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
            'sec': 'proximal',
            'groupSynMechs': PVSynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        #  TC -> E: proximal
        netParams.subConnParams['TC->E'] = {
            'preConds': {'cellType': ['TC', 'HTC']}, 
            'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
            'sec': 'proximal',
            'groupSynMechs': ESynMech, 
            'density': 'uniform'} 

        #------------------------------------------------------------------------------
        #  TCM -> E: apical
        netParams.subConnParams['TCM->E'] = {
            'preConds': {'cellType': ['TCM']}, 
            'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
            'sec': 'apic',
            'groupSynMechs': ESynMech, 
            'density': 'uniform'}
            

    #------------------------------------------------------------------------------
    # Background inputs 
    #------------------------------------------------------------------------------  
    if cfg.addBkgConn:
        # add bkg sources for E and I cells
        netParams.stimSourceParams['excBkg'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['exc'], 'noise': cfg.noiseBkg, 'number': 1e9}
        netParams.stimSourceParams['inhBkg'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['inh'], 'noise': cfg.noiseBkg, 'number': 1e9}
        
        if cfg.cochlearThalInput:
            from input import cochlearInputSpikes
            numCochlearCells = cfg.cochlearThalInput['numCells']
            cochlearSpkTimes = cochlearInputSpikes(numCells = numCochlearCells,
                                                duration = cfg.duration,
                                                freqRange = cfg.cochlearThalInput['freqRange'],
                                                toneFreq=cfg.cochlearThalInput['toneFreq'],
                                                loudnessDBs=cfg.cochlearThalInput['loudnessDBs'])
                                                
            netParams.popParams['cochlea'] = {'cellModel': 'VecStim', 'numCells': numCochlearCells, 'spkTimes': cochlearSpkTimes, 'ynormRange': layer['cochlear']}

        if cfg.ICThalInput:
            # load file with IC output rates
            from scipy.io import loadmat
            import numpy as np

            data = loadmat(cfg.ICThalInput['file'])
            fs = data['RsFs'][0][0]
            ICrates = data['BE_sout_population'].tolist()
            ICtimes = list(np.arange(0, cfg.duration, 1000./fs))  # list with times to set each time-dep rate
            
            ICrates = ICrates * 4 # 200 cells
            
            numCells = len(ICrates)

            # Option 1: create population of DynamicNetStims with time-varying rates
            #netParams.popParams['IC'] = {'cellModel': 'DynamicNetStim', 'numCells': numCells, 'ynormRange': layer['cochlear'],
            #    'dynamicRates': {'rates': ICrates, 'times': ICtimes}}

            # Option 2:
            from input import inh_poisson_generator
            
            maxLen = min(len(ICrates[0]), len(ICtimes))
            spkTimes = [[x+cfg.ICThalInput['startTime'] for x in inh_poisson_generator(ICrates[i][:maxLen], ICtimes[:maxLen], cfg.duration, cfg.ICThalInput['seed']+i)] for i in range(len(ICrates))]
            netParams.popParams['IC'] = {'cellModel': 'VecStim', 'numCells': numCells, 'ynormRange': layer['cochlear'],
                'spkTimes': spkTimes}


        # excBkg/I -> thalamus + cortex
        with open('cells/bkgWeightPops.json', 'r') as f:
            weightBkg = json.load(f)
        pops = list(cfg.allpops)
        pops.remove('IC')

        for pop in ['TC', 'TCM', 'HTC']:
            weightBkg[pop] *= cfg.EbkgThalamicGain 

        for pop in ['IRE', 'IREM', 'TI', 'TIM']:
            weightBkg[pop] *= cfg.IbkgThalamicGain 


        for pop in pops:
            netParams.stimTargetParams['excBkg->'+pop] =  {
                'source': 'excBkg', 
                'conds': {'pop': pop},
                'sec': 'apic', 
                'loc': 0.5,
                'synMech': ESynMech,
                'weight': weightBkg[pop],
                'synMechWeightFactor': cfg.synWeightFractionEE,
                'delay': cfg.delayBkg}

            netParams.stimTargetParams['inhBkg->'+pop] =  {
                'source': 'inhBkg', 
                'conds': {'pop': pop},
                'sec': 'proximal',
                'loc': 0.5,
                'synMech': 'GABAA',
                'weight': weightBkg[pop],
                'delay': cfg.delayBkg}

        # cochlea -> thal
        if cfg.cochlearThalInput:
            netParams.connParams['cochlea->ThalE'] = { 
                'preConds': {'pop': 'cochlea'}, 
                'postConds': {'cellType': ['TC', 'HTC']},
                'sec': 'soma', 
                'loc': 0.5,
                'synMech': ESynMech,
                'probability': cfg.probInput['ThalE'], 
                'weight': cfg.weightInput['ThalE'],
                'synMechWeightFactor': cfg.synWeightFractionEE,
                'delay': cfg.delayBkg}
            
            netParams.connParams['cochlea->ThalI'] = { 
                'preConds': {'pop': 'cochlea'}, 
                'postConds': {'cellType': ['RE']},
                'sec': 'soma', 
                'loc': 0.5,
                'synMech': ESynMech,
                'probability': cfg.probInput['ThalI'], 
                'weight': cfg.weightInput['ThalI'],
                'synMechWeightFactor': cfg.synWeightFractionEI,
                'delay': cfg.delayBkg}  

        # cochlea/IC -> thal
        if cfg.ICThalInput:
            netParams.connParams['IC->ThalE'] = { 
                'preConds': {'pop': 'IC'}, 
                'postConds': {'cellType': ['TC', 'HTC']},
                'sec': 'soma', 
                'loc': 0.5,
                'synMech': ESynMech,
                'probability': cfg.ICThalInput['probE'],
                'weight': cfg.ICThalInput['weightE'],
                'synMechWeightFactor': cfg.synWeightFractionEE,
                'delay': cfg.delayBkg}
            
            netParams.connParams['IC->ThalI'] = { 
                'preConds': {'pop': 'IC'}, 
                'postConds': {'cellType': ['RE', 'TI']},
                'sec': 'soma', 
                'loc': 0.5,
                'synMech': 'GABAA',
                'probability': cfg.ICThalInput['probI'],
                'weight': cfg.ICThalInput['weightI'],
                'delay': cfg.delayBkg}  


    #------------------------------------------------------------------------------
    # Current inputs (IClamp)
    #------------------------------------------------------------------------------
    # if cfg.addIClamp:
    #  	for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
    # 		params = getattr(cfg, key, None)
    # 		[pop,sec,loc,start,dur,amp] = [params[s] for s in ['pop','sec','loc','start','dur','amp']]
            
    #         		# add stim source
    # 		netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}
            
    # 		# connect stim source to target
    # 		netParams.stimTargetParams[key+'_'+pop] =  {
    # 			'source': key, 
    # 			'conds': {'pop': pop},
    # 			'sec': sec, 
    # 			'loc': loc}

    #------------------------------------------------------------------------------
    # NetStim inputs (to simulate short external stimuli; not bkg)
    #------------------------------------------------------------------------------
    if cfg.addNetStim:
        for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
            params = getattr(cfg, key, None)
            [pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
            [params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']] 

            # add stim source
            netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

            # connect stim source to target 
            netParams.stimTargetParams[key+'_'+pop] =  {
                'source': key, 
                'conds': {'pop': pop, 'ynorm': ynorm},
                'sec': sec, 
                'loc': loc,
                'synMech': synMech,
                'weight': weight,
                'synMechWeightFactor': synMechWeightFactor,
                'delay': delay}


    densities = netParams.popParams[pop]['density']

    #------------------------------------------------------------------------------
    # Description
    #------------------------------------------------------------------------------

    # netParams.description = """
    # v7 - Added template for connectivity
    # v8 - Added cell types
    # v9 - Added local connectivity
    # v10 - Added thalamic populations from prev model
    # v11 - Added thalamic conn from prev model
    # v12 - Added CT cells to L5B
    # v13 - Added CT cells to L5A
    # v14 - Fixed L5A & L5B E cell densities + added CT5A & CT5B to 'Epops'
    # v15 - Added cortical and thalamic conn to CT5A and CT5B 
    # v16 - Updated multiple cell types
    # v17 - Changed NGF -> I prob from strong (1.0) to weak (0.35)
    # v18 - Fixed bug in VIP cell morphology
    # v19 - Added in 2-compartment thalamic interneuron model 
    # v20 - Added TI conn and updated thal pop
    # v21 - Added exc+inh bkg inputs specific to each cell type
    # v22 - Made exc+inh bkg inputs specific to each pop; automated calculation
    # v23 - IE/II specific layer gains and simplified code (assume 'Allen_custom')
    # v24 - Fixed bug in IE/II specific layer gains
    # v25 - Fixed subconnparams TC->E and NGF1->E; made IC input deterministic
    # v26 - Changed NGF AMPA:NMDA ratio 
    # v27 - Split thalamic interneurons into core and matrix (TI and TIM)
    # v28 - Set recurrent TC->TC conn to 0
    # v29 - Added EI specific layer gains
    # v30 - Added EE specific layer gains; and split combined L1-3 gains into L1,L2,L3
    # v31 - Added EI postsyn-cell-type specific gains; update ITS4 and NGF
    # v32 - Added IE presyn-cell-type specific gains
    # v33 - Fixed bug in matrix thalamocortical conn (were very low)
    # v34 - Added missing conn from cortex to matrix thalamus IREM and TIM
    # """

    #------------------------------------------------------------------------------
    # Run
    #------------------------------------------------------------------------------
        
    sim.createSimulateAnalyze(netParams = netParams, simConfig = cfg)
        

if __name__ == '__main__':
    main()