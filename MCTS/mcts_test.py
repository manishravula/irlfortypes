import numpy as np
import mcts
import UNIverse as univ
from graph_tool.all import *

def draw_graph(magent):
    g = magent.graph
    vprops = {'label': g.vp.label, 'fillcolor':g.vp.color, 'overlap':'prism1000','overlap_scaling':'30'}
    gdraw = graphviz_draw(g,vprops=vprops)



universe = univ.universe()
magent = mcts.mcts(universe,1,'test',visualize=False)
for i in range(4):
    magent.rollout(universe.state)
# magent.visualize_props()
# draw_graph(magent)


