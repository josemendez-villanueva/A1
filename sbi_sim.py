from netpyne.network.pop import Pop
from netpyne.specs import netParams
import torch 
import numpy as np
from netpyne import specs, sim
import time
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.analysis.plot import pairplot


import sbi_test 

# Initial CFG, Needs to be defined first

sbi_test.cfg.duration = 1500 #should be 1500
sbi_test.cfg.printPopAvgRates = True #[[500, 750], [750, 1000], [1000, 1250], [1250, 1500]]
sbi_test.cfg.dt = 0.05
sbi_test.cfg.scaleDensity = 0.5

# plotting and saving params

sbi_test.cfg.analysis['plotTraces']['timeRange']= [500,1500]
sbi_test.cfg.analysis['plotTraces']['timeRange']= [500,1500]
sbi_test.cfg.analysis['plotTraces']['oneFigPer'] = 'trace'

sbi_test.cfg.saveCellSecs = False
sbi_test.cfg.saveCellConns = False


##Test chunk in order to get info from sbi_complex


# b = np.linspace(sbi_test.pop, sbi_test.pop*len(sbi_test.netParams.popParams), len(sbi_test.netParams.popParams))
b = 42

# Exc pops


sbi_test.Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']


## Inh pops 
sbi_test.Ipops = ['NGF1',                            # L1
                'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
                'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
                'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
                'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
                'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
                'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
                'IRE', 'IREM', 'TI']  # Thal 




pop_order = ['NGF1','IT2','SOM2','PV2','VIP2','NGF2','IT3','SOM3','PV3', 'VIP3','NGF3','ITP4' ,'ITS4','SOM4' ,'PV4','VIP4','NGF4',
                'IT5A','CT5A','SOM5A','PV5A','VIP5A','NGF5A','IT5B','CT5B','PT5B','SOM5B','PV5B','VIP5B','NGF5B','IT6','CT6','SOM6',
                'PV6','VIP6','NGF6','TC','TCM','HTC','IRE','IREM','TI']


dens_list = [] # This list will be for the histogram
density = 0
for pop in pop_order:
    den = sbi_test.netParams.popParams[pop]['density']
    density += den
    dens_list.append(density)



# Need to define Epops and Ipops or else stim do not get passed
def simulator(param):
    params = np.asarray(param)

    #bkg inputs
    #parameters that will be explored: weights that will be looked into to try and match to the pop rates of 42/43 populations
    sbi_test.cfg.EEGain = params[0]
    sbi_test.cfg.EIGain = params[1]

    sbi_test.cfg.IELayerGain['1'] = params[2]
    sbi_test.cfg.IELayerGain['2'] = params[2]
    sbi_test.cfg.IELayerGain['3'] = params[2]
    sbi_test.cfg.IELayerGain['4'] = params[3]
    sbi_test.cfg.IELayerGain['5'] = params[4]
    sbi_test.cfg.IELayerGain['6'] = params[5]

    sbi_test.cfg.IILayerGain['1'] = params[6]
    sbi_test.cfg.IILayerGain['2'] = params[6]
    sbi_test.cfg.IILayerGain['3'] = params[6]
    sbi_test.cfg.IILayerGain['4'] = params[7]
    sbi_test.cfg.IILayerGain['5'] = params[8]
    sbi_test.cfg.IILayerGain['6'] = params[9]

    sbi_test.cfg.thalamoCorticalGain = params[10] 
    sbi_test.cfg.intraThalamicGain = params[11]
    sbi_test.cfg.corticoThalamicGain = params[12]

    #Runs the simulation with the above given parameters
    sbi_test.run()

    simdata = sim.allSimData['popRates']


    # poprates = np.zeros(len(sbi_test.netParams.popParams))
    # i = 0
    # for list_values in simdata.values(): #Gets the poprates list for 500-750,750-1000 and so on....
    #     temp_poprates = 0
    #     for values in list_values: #the poprate values of the above list
    #         temp_poprates += values       
    #     poprates[i] = temp_poprates / 4 #Takes the average poprate within the distance so divide by the split
    #     i += 1


    poprates = np.zeros(len(sbi_test.netParams.popParams))
    j = 0
    for i in simdata.values():
        poprates[j] = (i)
 
        j += 1

    plotraces = np.array(sim.simData['V_soma']['cell_0'])
    time = np.array(sim.simData['t'])
    hist = np.histogram(poprates, dens_list)[0] #replaces the need for a summary statistics class/fitness function class
    return dict(stats = hist, time = time, pop = poprates, traces = plotraces)



# simulator([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

#Simulation Wrapper: Uses only histogram statistics (Can be fitted with fitness (potentially after checking integration with this model))
def simulation_wrapper(params):
    obs = simulator(params)
    summstats = torch.as_tensor(obs['stats'])
    return summstats


# simulation_wrapper([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
# Target Values will be set in the way that they are initialized on 'netParams.py'
pop_target = [10, # Layer 1
                5, 10, 10, 10, 10, # Layer 2
                5, 10, 10, 10, 10, # Layer 3
                5, 5, 10, 10, 10, 10, # Layer 4
                5, 5, 10, 10, 10, 10, # Layer 5A
                5, 5, 5, 10, 10, 10, 10, # Layer 5B
                5, 5, 10, 10, 10, 10, # Layer 6
                5, 5, 5, 10, 10, 10] #Thalamic pops not including 'TIM'
                
observable_baseline_stats = torch.as_tensor(np.histogram(pop_target, b)[0])


#Prior distribution Setup
prior_min = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
prior_max = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

#Unifrom Distribution setup 
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))

#Choose the option of running single-round or multi-round inference
inference_type = 'single'

if inference_type == 'single':
    posterior = infer(simulation_wrapper, prior, method='SNPE', 
                    num_simulations=1000, num_workers=8)
    samples = posterior.sample((10000,),
                                x = observable_baseline_stats)
    posterior_sample = posterior.sample((1,),
                                            x = observable_baseline_stats).numpy()

# elif inference_type == 'multi':
#     #Number of rounds that you want to run your inference
#     num_rounds = 2
#     #Driver for the multi-rounds inference
#     for _ in range(num_rounds):
#         posterior = infer(simulation_wrapper, prior, method='SNPE', 
#                     num_simulations=15000, num_workers=56)
#         prior = posterior.set_default_x(observable_baseline_stats)
#         samples = posterior.sample((10000,), x = observable_baseline_stats)

#     posterior_sample = posterior.sample((1,),
#                         x = observable_baseline_stats).numpy()

# else:
#     print('Wrong Input for Inference Type')

# # Plot Observed and Posterior

# #Gives the optimized paramters Here

# op_param = posterior_sample[0]

# x = simulator(op_param)
# t = x['time']

# #How to compare the poprates plot traces to the estimated one? Since we gave target one we cannot really do this since we do not have the target parameters


# print('Posterior Sample Param:', op_param)
# print('Pop Rate Estimates:', x['pop'])