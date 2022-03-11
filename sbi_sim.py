# from netpyne.network.pop import Pop
# from netpyne.specs import netParams
# import torch 
# import numpy as np
# from netpyne import specs, sim
# import time
# import json

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from sbi import utils as utils
# from sbi import analysis as analysis
# from sbi.inference.base import infer
# from sbi.analysis.plot import pairplot



# import sbi_test

# # Defaults parameters to one in order to get information to set-up density list = dens_list
# sbi_test.main()


# pop_order = ['NGF1','IT2','SOM2','PV2','VIP2','NGF2','IT3','SOM3','PV3', 'VIP3','NGF3','ITP4' ,'ITS4','SOM4' ,'PV4','VIP4','NGF4',
#             'IT5A','CT5A','SOM5A','PV5A','VIP5A','NGF5A','IT5B','CT5B','PT5B','SOM5B','PV5B','VIP5B','NGF5B','IT6','CT6','SOM6',
#             'PV6','VIP6','NGF6','TC','TCM','HTC','IRE','IREM','TI']


# # dens_list = [] # This list will be for the histogram

# density = 0
# dens_list = [] # This list will be for the histogram
# for pop in pop_order:
#     den = sbi_test.netParams.popParams[pop]['density']
#     density += den
#     dens_list.append(density)



# ## This is working model
# # def simulator(param):


# #     params = np.asarray(param)

# #     #Runs the simulation with the simulation-based parameters
# #     sbi_test.main(params)

# #     # sbi_sbi_test.run()

# #     data = sim.allSimData['popRates']

# #     simdata = data

# #     # Done to verify that it doesn't get used for all parallelized processes
# #     # del data
# #     # del sim.allSimData

# #     poprates = np.zeros(len(dens_list))
# #     j = 0
# #     for i in simdata.values():
# #         poprates[j] = (i)
# #         j += 1


# #     #Histogram uses poprates for given population-density
# #     #Depending on accuracy, good give a better istogram/summary statistic. This given histogram worked well on dummy network model.
# #     plotraces = np.array(sim.simData['V_soma']['cell_0'])
# #     time = np.array(sim.simData['t'])
# #     hist = np.histogram(poprates, dens_list)[0] #replaces the need for a summary statistics class/fitness function class


# #     print(params)
# #     print(poprates)

    
#     # return dict(stats = hist, time = time, pop = poprates, traces = plotraces)



# import cv2

# #### This simulatiuon will be assuming we use an embedding net and changing out to flattened image array of 32*32 = 1024
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn 
# import torch.nn.functional as F 
# from sbi import utils
# from sbi import analysis
# from sbi import inference
# from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
# import numpy as np

# import math

# #Simulator two

# # def simulator_custom(param):


# #     params = np.asarray(param)

# #     #Runs the simulation with the simulation-based parameters
# #     sbi_test.main(params)

# #     # sbi_sbi_test.run()

# #     data = sim.allSimData['popRates']

# #     simdata = data

# #     # Done to verify that it doesn't get used for all parallelized processes
# #     # del data
# #     # del sim.allSimData

# #     poprates = np.zeros(len(dens_list))
# #     j = 0
# #     for i in simdata.values():
# #         poprates[j] = (i)
# #         j += 1

# #     poprates = np.array(poprates)

# #     plt.plot(poprates)
# #     plt.savefig('pop_rate_image.png', dpi = 100)

# #     img = cv2.imread('pop_rate_image.png')

# #     import os
# #     os.remove('./pop_rate_image.png')

# #     #To make it 100*100*1 instead of 100*100*3
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #     images = cv2.resize(gray, (32, 32))
# #     images = cv2.normalize(images, None, 0, 1, cv2.NORM_MINMAX)

# #     images = images.reshape(1,-1)

# #     print(np.shape(images))



# ###############################     return images



# ################Simulator three


# def tabulate(x, y, f):
#     return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

# def cos_sum(a, b):
#     return(math.cos(a+b))



# def transform(poprates):
#     """Compute the Gramian Angular Field of an image"""
#     # Min-Max scaling
#     min_ = np.amin(poprates)
#     max_ = np.amax(poprates)
#     scaled_serie = (2*poprates - max_ - min_)/(max_ - min_)

#     # Floating point inaccuracy!
#     scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
#     scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

#     # Polar encoding
#     phi = np.arccos(scaled_serie)
#     # Note! The computation of r is not necessary
#     r = np.linspace(0, 1, len(scaled_serie))

#     # GAF Computation (every term of the matrix)
#     gaf = tabulate(phi, phi, cos_sum)


#     gaf = gaf.reshape(1,-1)


#     return gaf




# def simulator_custom(param):


#     params = np.asarray(param)

#     #Runs the simulation with the simulation-based parameters
#     sbi_test.main(params)

#     # sbi_sbi_test.run()

#     data = sim.allSimData['popRates']

#     simdata = data

#     # Done to verify that it doesn't get used for all parallelized processes
#     # del data
#     # del sim.allSimData

#     poprates = np.zeros(len(dens_list))
#     j = 0
#     for i in simdata.values():
#         poprates[j] = (i)
#         j += 1



#     return transform(poprates)








# ##############################






# # simulate samples
# true_parameter = [10, # Layer 1
#                 5, 10, 10, 10, 10, # Layer 2
#                 5, 10, 10, 10, 10, # Layer 3
#                 5, 5, 10, 10, 10, 10, # Layer 4
#                 5, 5, 10, 10, 10, 10, # Layer 5A
#                 5, 5, 5, 10, 10, 10, 10, # Layer 5B
#                 5, 5, 10, 10, 10, 10, # Layer 6
#                 5, 5, 5, 10, 10, 10] #Thalamic pops not including 'TIM')




# # plt.plot(true_parameter)
# # plt.savefig('pop_rate_image.png', dpi = 100)

# # img = cv2.imread('pop_rate_image.png')

# # import os
# # os.remove('./pop_rate_image.png')

# # #To make it 100*100*1 instead of 100*100*3
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # images = cv2.resize(gray, (32, 32))
# # images = cv2.normalize(images, None, 0, 1, cv2.NORM_MINMAX)

# # images = images.reshape(1,-1)

# # x_observed = images


# x_observed = transform(true_parameter)









# class SummaryNet(nn.Module): 

#     def __init__(self): 
#         super().__init__()
#         # 2D convolutional layer
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
#         # Maxpool layer that reduces 32x32 image to 4x4
#         self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
#         # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
#         self.fc = nn.Linear(in_features=6*4*4, out_features=8) 

#     def forward(self, x):
#         x = x.view(-1, 1, 84, 84)
#         x = self.pool(F.relu(self.conv1(x)))
#         x = x.view(-1, 6*4*4)
#         x = F.relu(self.fc(x))
#         return x

# embedding_net = SummaryNet()



# #Prior distribution Setup
# prior_min = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
# prior_max = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# #Unifrom Distribution setup 
# prior = utils.BoxUniform(low=torch.as_tensor(prior_min), 
#                                     high=torch.as_tensor(prior_max))


# # make a SBI-wrapper on the simulator object for compatibility
# simulator_wrapper, prior = prepare_for_sbi(simulator_custom, prior)

# # instantiate the neural density estimator
# neural_posterior = utils.posterior_nn(model='maf', 
#                                       embedding_net=embedding_net,
#                                       hidden_features=10,
#                                       num_transforms=2)

# # setup the inference procedure with the SNPE-C procedure
# inference = SNPE(prior=prior, density_estimator=neural_posterior)

# # run the inference procedure on one round and 10000 simulated data points
# theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=1000, num_workers=4)
# density_estimator = inference.append_simulations(theta, x).train()
# posterior = inference.build_posterior(density_estimator)















# # End of this process
# ########

# ## Working Model
# # #Simulation Wrapper: Uses only histogram statistics (Can be fitted with fitness (potentially after checking integration with this model))
# # def simulation_wrapper(params):
# #     obs = simulator(params)
# #     summstats = torch.as_tensor(obs['stats'])
# #     return summstats

# # simulation_wrapper([1.2641431, 1.2722495, 1.1157576, 1.3192368, 1.1917353, 0.9152887, 1.4060045,
# #  1.1527985, 1.9112793, 0.5862609, 1.1985906, 1.584882,  1.0767709])


# # # # Target Values will be set in the way that they are initialized on 'netParams.py'
# # # pop_target = [10, # Layer 1
# # #                 5, 10, 10, 10, 10, # Layer 2
# # #                 5, 10, 10, 10, 10, # Layer 3
# # #                 5, 5, 10, 10, 10, 10, # Layer 4
# # #                 5, 5, 10, 10, 10, 10, # Layer 5A
# # #                 5, 5, 5, 10, 10, 10, 10, # Layer 5B
# # #                 5, 5, 10, 10, 10, 10, # Layer 6
# # #                 5, 5, 5, 10, 10, 10] #Thalamic pops not including 'TIM'
                
# # # observable_baseline_stats = torch.as_tensor(np.histogram(pop_target, dens_list)[0])


# # # #Prior distribution Setup
# # # prior_min = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
# # # prior_max = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# # # #Unifrom Distribution setup 
# # # prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
# # #                                     high=torch.as_tensor(prior_max))



# # # inference_type = 'single'


# # # if inference_type == 'single':
# # #     posterior = infer(simulation_wrapper, prior, method='SNPE', 
# # #                     num_simulations=1000, num_workers=6)
# # #     samples = posterior.sample((1000,),
# # #                                 x = observable_baseline_stats)
# # #     posterior_sample = posterior.sample((1,),
# # #                                             x = observable_baseline_stats).numpy()



# # # op_param = posterior_sample[0]

# # # x = simulator(op_param)
# # # print(op_param)
# # # # t = x['time']





########## Test of the new file structure
###########
###########3
############
############




from netpyne.network.pop import Pop
from netpyne.specs import netParams
import torch 
import numpy as np
from netpyne import specs, sim

from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.analysis.plot import pairplot

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from sbi import utils
from sbi import analysis
from sbi import inference
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
import numpy as np

import math
import cv2



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



## This is working model

flag = 1
flag_two = 5

if flag == 0:

    def simulator(param):


        params = np.asarray(param)

        #Runs the simulation with the simulation-based parameters
        sbi_test.main(params)

        # sbi_sbi_test.run()

        data = sim.allSimData['popRates']

        simdata = data

        # Done to verify that it doesn't get used for all parallelized processes
        # del data
        # del sim.allSimData

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


        print(params)
        print(poprates)

        
        return dict(stats = hist, time = time, pop = poprates, traces = plotraces)



    # Working Model
    #Simulation Wrapper: Uses only histogram statistics (Can be fitted with fitness (potentially after checking integration with this model))
    def simulation_wrapper(params):
        obs = simulator(params)
        summstats = torch.as_tensor(obs['stats'])
        return summstats

    # simulation_wrapper([1.2641431, 1.2722495, 1.1157576, 1.3192368, 1.1917353, 0.9152887, 1.4060045,
    #  1.1527985, 1.9112793, 0.5862609, 1.1985906, 1.584882,  1.0767709])


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





#### This simulatiuon will be assuming we use an embedding net and changing out to flattened image array of 32*32 = 1024

else:

    if flag_two == 0:


        def tabulate(x, y, f):
            return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

        def cos_sum(a, b):
            return(math.cos(a+b))



        def transform(poprates):
            """Compute the Gramian Angular Field of an image"""
            # Min-Max scaling
            min_ = np.amin(poprates)
            max_ = np.amax(poprates)
            scaled_serie = (2*poprates - max_ - min_)/(max_ - min_)

            # Floating point inaccuracy!
            scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
            scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

            # Polar encoding
            phi = np.arccos(scaled_serie)
            # Note! The computation of r is not necessary
            r = np.linspace(0, 1, len(scaled_serie))

            # GAF Computation (every term of the matrix)
            gaf = tabulate(phi, phi, cos_sum)


            gaf = gaf.reshape(1,-1)


            return gaf




        def simulator_custom(param):


            params = np.asarray(param)

            #Runs the simulation with the simulation-based parameters
            sbi_test.main(params)

            # sbi_sbi_test.run()

            data = sim.allSimData['popRates']

            simdata = data

            # Done to verify that it doesn't get used for all parallelized processes
            # del data
            # del sim.allSimData

            poprates = np.zeros(len(dens_list))
            j = 0
            for i in simdata.values():
                poprates[j] = (i)
                j += 1



            return transform(poprates)




        # simulate samples
        true_parameter = [10, # Layer 1
                        5, 10, 10, 10, 10, # Layer 2
                        5, 10, 10, 10, 10, # Layer 3
                        5, 5, 10, 10, 10, 10, # Layer 4
                        5, 5, 10, 10, 10, 10, # Layer 5A
                        5, 5, 5, 10, 10, 10, 10, # Layer 5B
                        5, 5, 10, 10, 10, 10, # Layer 6
                        5, 5, 5, 10, 10, 10] #Thalamic pops not including 'TIM')


        x_observed = transform(true_parameter)





    elif flag_two == 1:
        
        def simulator_custom(param):


            params = np.asarray(param)

            #Runs the simulation with the simulation-based parameters
            sbi_test.main(params)

            # sbi_sbi_test.run()

            data = sim.allSimData['popRates']

            simdata = data

            # Done to verify that it doesn't get used for all parallelized processes
            # del data
            # del sim.allSimData

            poprates = np.zeros(len(dens_list))
            j = 0
            for i in simdata.values():
                poprates[j] = (i)
                j += 1

            poprates = np.array(poprates)

            plt.plot(poprates)
            plt.savefig('pop_rate_image.png', dpi = 100)

            img = cv2.imread('pop_rate_image.png')

            import os
            os.remove('./pop_rate_image.png')

            #To make it 100*100*1 instead of 100*100*3
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            images = cv2.resize(gray, (32, 32))
            images = cv2.normalize(images, None, 0, 1, cv2.NORM_MINMAX)

            images = images.reshape(1,-1)

            print(np.shape(images))

            return images

        # simulate samples
        true_parameter = [10, # Layer 1
                        5, 10, 10, 10, 10, # Layer 2
                        5, 10, 10, 10, 10, # Layer 3
                        5, 5, 10, 10, 10, 10, # Layer 4
                        5, 5, 10, 10, 10, 10, # Layer 5A
                        5, 5, 5, 10, 10, 10, 10, # Layer 5B
                        5, 5, 10, 10, 10, 10, # Layer 6
                        5, 5, 5, 10, 10, 10] #Thalamic pops not including 'TIM')


        plt.plot(true_parameter)
        plt.savefig('pop_rate_image.png', dpi = 100)

        img = cv2.imread('pop_rate_image.png')

        import os
        os.remove('./pop_rate_image.png')

        #To make it 100*100*1 instead of 100*100*3
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images = cv2.resize(gray, (32, 32))
        images = cv2.normalize(images, None, 0, 1, cv2.NORM_MINMAX)

        images = images.reshape(1,-1)

        x_observed = images




    else:



        def simulator_custom(param):


            params = np.asarray(param)

            #Runs the simulation with the simulation-based parameters
            sbi_test.main(params)

            # sbi_sbi_test.run()

            data = sim.allSimData['popRates']

            simdata = data

            # Done to verify that it doesn't get used for all parallelized processes
            # del data
            # del sim.allSimData

            poprates = np.zeros(len(dens_list))
            j = 0
            for i in simdata.values():
                poprates[j] = (i)
                j += 1


            pop = [[poprates[0],0,0,0,0,0,0,0],

                        [0,poprates[1],0,0,0,0,0,0],
                        [0,poprates[2],0,0,0,0,0,0],
                        [0,poprates[3],0,0,0,0,0,0],
                        [0,poprates[4],0,0,0,0,0,0],
                        [0,poprates[5],0,0,0,0,0,0],

                        [0,0,poprates[6],0,0,0,0,0],
                        [0,0,poprates[7],0,0,0,0,0],
                        [0,0,poprates[8],0,0,0,0,0],
                        [0,0,poprates[9],0,0,0,0,0],
                        [0,0,poprates[10],0,0,0,0,0],


                        [0,0,0,poprates[11],0,0,0,0],
                        [0,0,0,poprates[12],0,0,0,0],
                        [0,0,0,poprates[13],0,0,0,0],
                        [0,0,0,poprates[14],0,0,0,0],
                        [0,0,0,poprates[15],0,0,0,0],
                        [0,0,0,poprates[16],0,0,0,0],

                        [0,0,0,0,poprates[17],0,0,0],
                        [0,0,0,0,poprates[18],0,0,0],
                        [0,0,0,0,poprates[19],0,0,0],
                        [0,0,0,0,poprates[20],0,0,0],
                        [0,0,0,0,poprates[21],0,0,0],
                        [0,0,0,0,poprates[22],0,0,0],

                        [0,0,0,0,0,poprates[23],0,0],
                        [0,0,0,0,0,poprates[24],0,0],
                        [0,0,0,0,0,poprates[25],0,0],
                        [0,0,0,0,0,poprates[26],0,0],
                        [0,0,0,0,0,poprates[27],0,0],
                        [0,0,0,0,0,poprates[28],0,0],
                        [0,0,0,0,0,poprates[29],0,0],

                        [0,0,0,0,0,0,poprates[30],0],
                        [0,0,0,0,0,0,poprates[31],0],
                        [0,0,0,0,0,0,poprates[32],0],
                        [0,0,0,0,0,0,poprates[33],0],
                        [0,0,0,0,0,0,poprates[34],0],
                        [0,0,0,0,0,0,poprates[35],0],

                        [0,0,0,0,0,0,0,poprates[36]],
                        [0,0,0,0,0,0,0,poprates[37]],
                        [0,0,0,0,0,0,0,poprates[38]],
                        [0,0,0,0,0,0,0,poprates[39]],
                        [0,0,0,0,0,0,0,poprates[40]],
                        [0,0,0,0,0,0,0,poprates[41]]]


            pop = np.reshape(pop, (1,-1))


            return pop




        # simulate samples
        true_parameter = [[10,0,0,0,0,0,0,0],

                            [0,5,0,0,0,0,0,0],
                            [0,10,0,0,0,0,0,0],
                            [0,10,0,0,0,0,0,0],
                            [0,10,0,0,0,0,0,0],
                            [0,10,0,0,0,0,0,0],

                            [0,0,5,0,0,0,0,0],
                            [0,0,10,0,0,0,0,0],
                            [0,0,10,0,0,0,0,0],
                            [0,0,10,0,0,0,0,0],
                            [0,0,10,0,0,0,0,0],


                            [0,0,0,5,0,0,0,0],
                            [0,0,0,5,0,0,0,0],
                            [0,0,0,10,0,0,0,0],
                            [0,0,0,10,0,0,0,0],
                            [0,0,0,10,0,0,0,0],
                            [0,0,0,10,0,0,0,0],

                            [0,0,0,0,5,0,0,0],
                            [0,0,0,0,5,0,0,0],
                            [0,0,0,0,10,0,0,0],
                            [0,0,0,0,10,0,0,0],
                            [0,0,0,0,10,0,0,0],
                            [0,0,0,0,10,0,0,0],

                            [0,0,0,0,0,5,0,0],
                            [0,0,0,0,0,5,0,0],
                            [0,0,0,0,0,5,0,0],
                            [0,0,0,0,0,10,0,0],
                            [0,0,0,0,0,10,0,0],
                            [0,0,0,0,0,10,0,0],
                            [0,0,0,0,0,10,0,0],

                            [0,0,0,0,0,0,5,0],
                            [0,0,0,0,0,0,5,0],
                            [0,0,0,0,0,0,10,0],
                            [0,0,0,0,0,0,10,0],
                            [0,0,0,0,0,0,10,0],
                            [0,0,0,0,0,0,10,0],

                            [0,0,0,0,0,0,0,5],
                            [0,0,0,0,0,0,0,5],
                            [0,0,0,0,0,0,0,5],
                            [0,0,0,0,0,0,0,10],
                            [0,0,0,0,0,0,0,10],
                            [0,0,0,0,0,0,0,10]]

        x_observed = np.reshape(true_parameter, (1,-1))




    class SummaryNet(nn.Module): 

        def __init__(self): 
            super().__init__()
            # 2D convolutional layer
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
            # Maxpool layer that reduces 32x32 image to 4x4
            self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
            # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
            self.fc = nn.Linear(in_features=6*4*4, out_features=8) 

        def forward(self, x):
            x = x.view(-1, 1, 42, 8)
            x = self.pool(F.relu(self.conv1(x)))
            x = x.view(-1, 6*4*4)
            x = F.relu(self.fc(x))
            return x

    embedding_net = SummaryNet()



    #Prior distribution Setup
    prior_min = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
    prior_max = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    #Unifrom Distribution setup 
    prior = utils.BoxUniform(low=torch.as_tensor(prior_min), 
                                        high=torch.as_tensor(prior_max))


    # make a SBI-wrapper on the simulator object for compatibility
    simulator_wrapper, prior = prepare_for_sbi(simulator_custom, prior)

    # instantiate the neural density estimator
    neural_posterior = utils.posterior_nn(model='maf', 
                                        embedding_net=embedding_net,
                                        hidden_features=10,
                                        num_transforms=2)

    # setup the inference procedure with the SNPE-C procedure
    inference = SNPE(prior=prior, density_estimator=neural_posterior)

    # run the inference procedure on one round and 10000 simulated data points
    theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=1000, num_workers=4)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
