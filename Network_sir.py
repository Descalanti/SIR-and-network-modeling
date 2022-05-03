'''
Code for cleaning and plotting the Star Wars Survey Data from Fivethirtyeight. 
Authors: Devyn Escalanti & Franklin Castillo 
University of Central Florida 
Email: dtescalanti@gmail.com
'''

pip install EoN
import EoN
import matplotlib.pyplot as plt
import networkx          as nx
import numpy             as np
import pandas            as pd
import random

from collections  import defaultdict

def generate_graph( graph
                  , N                     = 100
                  , recovery_rate         = 0.62
                  , initial_infected_rate = 0.05
                  ):
  SIR_results        = {}
  probabilities      = np.arange(start=0.001, stop=0.100, step=0.01)
  transmission_rates = np.arange(start=1.3  , stop=2.7  , step=0.01)      # average 2.53

  for transmission_rate in transmission_rates:
    G = None

    if graph.__name__ == 'scale_free_graph':
      G       = nx.Graph(graph(N))
      SIR_run = 'g_{:.02f}'.format(transmission_rate)

      SIR_results[SIR_run] = EoN.fast_SIR( G
                                         , tau              = transmission_rate
                                         , gamma            = recovery_rate
                                         , rho              = initial_infected_rate
                                         , return_full_data = True
                                         , tmax             = 5000
                                         )

    if graph.__name__ == 'watts_strogatz_graph':
      for probability in probabilities:
        G       = nx.Graph(graph(n = N, k = 5, p = probability))
        SIR_run = 'g_{:.03f}_{:.02f}'.format(probability, transmission_rate)

        SIR_results[SIR_run] = EoN.fast_SIR( G
                                           , tau              = transmission_rate
                                           , gamma            = recovery_rate
                                           , rho              = initial_infected_rate
                                           , return_full_data = True
                                           , tmax             = 5000
                                           )

  return SIR_results

# Create four diffusion metrics
# 1. SIR with scale free network
# 2. SIR with small world network

SIR_scale_free  = generate_graph(nx.scale_free_graph)
SIR_small_world = generate_graph(nx.watts_strogatz_graph)

#plotting the scale-free and ebola simulation 
SIR_scale_free['g_2.15'].display(time = 2)
plt.show()
