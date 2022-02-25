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

# Defaults parameters to one in order to get information to set-up density list = dens_list
sbi_test.main()


pop_order = ['NGF1','IT2','SOM2','PV2','VIP2','NGF2','IT3','SOM3','PV3', 'VIP3','NGF3','ITP4' ,'ITS4','SOM4' ,'PV4','VIP4','NGF4',
            'IT5A','CT5A','SOM5A','PV5A','VIP5A','NGF5A','IT5B','CT5B','PT5B','SOM5B','PV5B','VIP5B','NGF5B','IT6','CT6','SOM6',
            'PV6','VIP6','NGF6','TC','TCM','HTC','IRE','IREM','TI']


# dens_list = [] # This list will be for the histogram

density = 0
dens_list = [] # This list will be for the histogram
for pop in pop_order:
    den = sbi_test.netParams.popParams[pop]['density']
    density += den
    dens_list.append(density)




def simulator(param):


    params = np.asarray(param)

    #Runs the simulation with the simulation-based parameters
    sbi_test.main(params)

    # sbi_sbi_test.run()

    data = sim.allSimData['popRates']

    simdata = data

    # Done to verify that it doesn't get used for all parallelized processes
    del data
    del sim.allSimData

    poprates = np.zeros(len(dens_list))
    j = 0
    for i in simdata.values():
        poprates[j] = (i)
        j += 1


    #Histogram uses poprates for given population-density
    #Depending on accuracy, good give a better istogram/summary statistic. This given histogram worked well on dummy network model.
    plotraces = np.array(sim.simData['V_soma']['cell_0'])
    time = np.array(sim.simData['t'])
    hist = np.histogram(poprates, dens_list)[0] #replaces the need for a summary statistics class/fitness function class


    # print(params)
    #print(poprates)

    
    return dict(stats = hist, time = time, pop = poprates, traces = plotraces)




#Simulation Wrapper: Uses only histogram statistics (Can be fitted with fitness (potentially after checking integration with this model))
def simulation_wrapper(params):
    obs = simulator(params)
    summstats = torch.as_tensor(obs['stats'])
    return summstats




# Target Values will be set in the way that they are initialized on 'netParams.py'
pop_target = [10, # Layer 1
                5, 10, 10, 10, 10, # Layer 2
                5, 10, 10, 10, 10, # Layer 3
                5, 5, 10, 10, 10, 10, # Layer 4
                5, 5, 10, 10, 10, 10, # Layer 5A
                5, 5, 5, 10, 10, 10, 10, # Layer 5B
                5, 5, 10, 10, 10, 10, # Layer 6
                5, 5, 5, 10, 10, 10] #Thalamic pops not including 'TIM'
                
observable_baseline_stats = torch.as_tensor(np.histogram(pop_target, dens_list)[0])


#Prior distribution Setup
prior_min = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
prior_max = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

#Unifrom Distribution setup 
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))



inference_type = 'single'


if inference_type == 'single':
    posterior = infer(simulation_wrapper, prior, method='SNPE', 
                    num_simulations=1000, num_workers=6)
    samples = posterior.sample((1000,),
                                x = observable_baseline_stats)
    posterior_sample = posterior.sample((1,),
                                            x = observable_baseline_stats).numpy()



op_param = posterior_sample[0]

x = simulator(op_param)
print(op_param)
# t = x['time']
