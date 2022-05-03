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


def sir_graph_metrics(sir_simulation):
    metrics = pd.DataFrame(columns = [ 'Average Degree Centrality'     
                                     , 'Average Betweenness Centrality'
                                     , 'Average Local Efficiency'      
                                     , 'Average Global Efficiency'
                                     ])
    
    for key in sir_simulation.keys():
        scale_free_graph = nx.Graph(sir_simulation[key].transmission_tree())

        metrics.loc[key] = [ '{:05f}'.format(np.mean(list(nx.degree_centrality(scale_free_graph))))
                           , '{:05f}'.format(np.mean(list(nx.betweenness_centrality(scale_free_graph))))
                           , '{:05f}'.format(nx.local_efficiency(scale_free_graph))
                           , '{:05f}'.format(nx.global_efficiency(scale_free_graph))
                           ]
    
    return metrics

#Generating graphs 
SIR_scale_free  = generate_graph(nx.scale_free_graph)
SIR_small_world = generate_graph(nx.watts_strogatz_graph)

#graphing scale free 
animation = SIR_scale_free['g_2.15'].animate()
animation.save('scale_free.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

#graphing small world 
animation = SIR_small_world['g_0.041_2.15'].animate()
animation.save('small_world.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

#scale free networks 
scale_free_graph = nx.Graph(SIR_scale_free['g_2.15'].transmission_tree())

metrics = { 'Average Degree Centrality'      : '{:05f}'.format(np.mean(list(nx.degree_centrality(scale_free_graph))))
          , 'Average Betweenness Centrality' : '{:05f}'.format(np.mean(list(nx.betweenness_centrality(scale_free_graph))))
          , 'Average Local Efficiency'       : '{:05f}'.format(nx.local_efficiency(scale_free_graph))
          , 'Average Global Efficiency'      : '{:05f}'.format(nx.global_efficiency(scale_free_graph))
          }

metrics

df_scale_free = sir_graph_metrics(SIR_scale_free)
len(df_scale_free)
display(df_scale_free)


#small world metrics
small_world_graph = nx.Graph(SIR_small_world['g_0.041_2.15'].transmission_tree())

metrics = { 'Average Degree Centrality'      : '{:05f}'.format(np.mean(list(nx.degree_centrality(small_world_graph))))
          , 'Average Betweenness Centrality' : '{:05f}'.format(np.mean(list(nx.betweenness_centrality(small_world_graph))))
          , 'Average Local Efficiency'       : '{:05f}'.format(nx.local_efficiency(small_world_graph))
          , 'Average Global Efficiency'      : '{:05f}'.format(nx.global_efficiency(small_world_graph))
          }

metrics

df_small_world = sir_graph_metrics(SIR_small_world)
len(df_small_world)
display(df_small_world)
