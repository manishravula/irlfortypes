import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import pdb

# RED = np.array([255,0,0])
# BLUE = np.array([0,0,255])
# GREEN = np.array([0,255,0])
# YELLOW = np.array([255,255,0])
# BLACK = np.array([0,0,0])
# CYAN = np.array((224,255,255))

RED = "#FF0000"
BLUE = "#1E90FF"
GREEN = "#7FFF00"
YELLOW = "#FFFF00"
BLACK = "#696969"
LINEN = "#FAF0E6"
BROWN = "#8B4513"

#todo:
# Add a update_function. We don't have to create new objects everytime we
# want to change end_goal or start_goal or some changes in the arena.
#NOT NEEDED. The creation of objects doesn't take much time.


class pos_node():
    def __init__(self,loc,h,g,parent,is_obstacle):
        self.loc = np.array(loc)
        self.h = h #distance to goal
        self.g = g #distance from start
        self.f = 0
        self.parent = parent
        self.is_obstacle = is_obstacle
        self.update_f()
        return

    def update_f(self):
        self.f = self.g + 1000*self.h
        return

    def __str__(self):
        main_str = ' loc: '+str(self.loc)+' | h: '+str(self.h)+' | g: '+str(self.g)+' | f: '+str(self.f)+' | is_obstacle '+str(self.is_obstacle)+' '
        if self.parent:
            return main_str+' | parent_loc: '+str(self.parent.loc)
        else:
            return main_str

class astar():
    def __init__(self,grid_value_matrix,start_pos,goal_pos,interactive=False,**kwargs):
        self.grid_value_matrix = grid_value_matrix
        self.grid_size = len(self.grid_value_matrix)
        self.start_pos = np.asarray(start_pos,int)
        self.goal_pos = np.asarray(goal_pos,int)
        self.movement_directions = kwargs.get('movement_directions',4)
        self.distance_metric = kwargs.get('distance_metric', 'euclidean')
        self.interactive = interactive
        self.manual_obstacles = manual_obstacles
        self.obstacle_pos = np.array(np.where(self.grid_value_matrix!=0)).T





        self.grid_matrix = self.createObjectGrid()
        self.init_hValues()

        self.start_node = self.grid_matrix[self.start_pos[0]][self.start_pos[1]]
        self.goal_node = self.grid_matrix[self.goal_pos[0]][self.goal_pos[1]]
        self.curr_node = self.start_node

        self.open_list = [self.curr_node]
        self.close_list = []

        if self.interactive:
            plt.ion()
            self.fig,self.ax = plt.subplots()
            grid = np.mgrid[0:self.grid_size,0:self.grid_size]
            x,y= grid[0],grid[1]

            color_mat = [[LINEN for i in range(self.grid_size)] for j in range(self.grid_size)]
            self.color_matrix = np.array(color_mat)
            self.color_matrix[self.goal_pos[0],self.goal_pos[1]]=BLACK
            self.color_matrix[self.start_pos[0],self.start_pos[1]]=BLACK
            for obs in self.obstacle_pos:
                self.color_matrix[obs[0]][obs[1]]=BROWN

            self.size_matrix = np.ones((self.grid_size,self.grid_size))*6000
            self.sc = plt.scatter(x,y,c=self.color_matrix.flatten(),s=self.size_matrix)
            plt.draw()


    def createObjectGrid(self):
        objectgrid = []
        for row,i in zip(self.grid_value_matrix,range(self.grid_size)):
            row_list = []
            for is_obstacle,j in zip(row,range(self.grid_size)):
                obj = pos_node((i,j),0,0,None,is_obstacle)
                row_list.append(obj)
            objectgrid.append(row_list)
        return objectgrid

    def init_hValues(self):
        """
        Precompute and store all the h-values to be used in the future.
        Currently uses Euclidean heuristic. Planned support for Manhattan and other types.
        :return:
        """
        for row in self.grid_matrix:
            for node in row:
                node.h = np.linalg.norm(node.loc-self.goal_pos)
                node.update_f()
        return

    def select_children(self,node):
        """
        Selects all non-closed, non-obstacle nodes according to
        movement_directions
        :return: list of copied (new-objects) children.
        """
        if self.movement_directions==8:
            diffs_x = np.array([-1,-1,-1,0,0,1,1,1])
            diffs_y = np.array([-1,0,1,-1,1,-1,0,1])

        else:
            diffs_x = np.array([-1,0,0,1])
            diffs_y = np.array([0,-1,1,0])

        diffs = np.vstack((diffs_x,diffs_y)).T
        neighbor_indices = node.loc + diffs

        #Checks if neighbors are inside the grid.
        sanity_check = np.all(np.array((neighbor_indices[:,0]>=0, neighbor_indices[:,1]>=0,neighbor_indices[:,0]<=self.grid_size-1,neighbor_indices[:,1]<=self.grid_size-1)),axis=0)

        inside_indices = neighbor_indices[sanity_check]
        #Only return neighbors which are not in the closed-list
        neighbors = [copy.copy(node) for node in [self.grid_matrix[insnode_loc[0]][insnode_loc[1]] for insnode_loc in inside_indices] if ((not node.is_obstacle) and (node not in self.close_list))]

        return neighbors

    def solve(self):
        """
        Main algorithmic body.
        :return: final curr_pos
        """

        while(True):

            #loop init stuff

            # pdb.set_trace()


            self.open_list.remove(self.curr_node)
            self.close_list.append(self.curr_node)
            self.children_array = []

            #Exploring around

            #Select all non-closed, non-obstacle nodes as children
            self.children_array = self.select_children(self.curr_node)
            #Update
            for child in self.children_array:
                child.parent = self.curr_node
                child.g = self.curr_node.g + np.linalg.norm(child.loc -self.curr_node.loc)
                child.update_f()

            if self.interactive:
                #Set the color of all open positions to green
                for node in self.open_list:
                    self.color_matrix[node.loc[0],node.loc[1]]=GREEN
                #set the color of all closed positions to blue
                for node in self.close_list:
                    self.color_matrix[node.loc[0],node.loc[1]]=BLUE
                #set the color of curr_position to red
                self.color_matrix[self.curr_node.loc[0],self.curr_node.loc[1]] = RED
                #set the color of children to yellow
                for node in self.children_array:
                    self.color_matrix[node.loc[0],node.loc[1]]= YELLOW
                self.sc.set_color(self.color_matrix.flatten())
                self.fig.canvas.draw_idle()
                plt.pause(.551)

            #Decision
            for child in self.children_array:
                # print("***")
                # print(child)
                # print("***")
                if np.all(child.loc==self.goal_pos):
                    #setting original goal object[one that is stored in self.grid_matrix]'s parent as curr_node
                    self.goal_node.parent = self.curr_node

                    #for returning purpose
                    self.curr_node = self.goal_node
                    return self.curr_node

                original_child_node = self.grid_matrix[child.loc[0]][child.loc[1]]
                if original_child_node in self.open_list:
                    if child.g>original_child_node.g:
                        #Is the path to a child through the current node
                        #worse than through whatever was its parent?
                        pass
                    else:
                        self.open_list.remove(original_child_node)
                        self.grid_matrix[child.loc[0]][child.loc[1]] = child
                        #original location editing now.

                        self.open_list.append(self.grid_matrix[child.loc[0]][child.loc[1]])

                else:
                    #The path to the child through the current node
                    #is better than through this node we are in right now.
                    #so change the parent of the child node (original one)
                    #to the curr_node now. And update the f values due to the
                    #changed g value. You can just copy the object
                    self.grid_matrix[child.loc[0]][child.loc[1]] = child
                    #original location editing now.
                    self.open_list.append(self.grid_matrix[child.loc[0]][child.loc[1]])



           # #Loop control
            if not self.open_list:
               #The list is empty. No trajectory found.
               self.curr_node = None
               return self.curr_node
            else:
                openFvalList = [node.f for node in self.open_list]
                self.curr_node = self.open_list[openFvalList.index(min(openFvalList))]



        return

    def retrace_path(self):
        path = []
        curr_node = self.curr_node
        while curr_node!=self.start_node:
            path.insert(0,curr_node)
            curr_node = curr_node.parent
        return path


    def find_minimumpath(self):
        sol = self.solve()
        if sol is None:
            print("No path to goal found")
            return []
        else:
            print("Path to goal found")
            return self.retrace_path()

#
#
# if __name__ == "__main__":
#
#     # grid_matrix = np.zeros((10,10))
#     # grid_matrix[1,1] = 1
#     grid_matrix = np.eye(10)
#     for i in range(3):
#         grid_matrix[i,i]=0
#
#
#     start_position = np.array([8,0])
#     goal_position = np.array([0,8])
#     ask_for_obstacles = False
#     interactive = True
#     time_array = []
#     for i in range(1):
#         start_position = np.array([8,np.random.randint(0,7)])
#         end_position = np.array([np.random.randint(0,7),8])
#
#         begin_time = time.time()
#         a = astar(grid_matrix,start_position,goal_position,False,True)
#         sol = a.find_minimumpath()
#         time_array.append(time.time()-begin_time)
#     #plt.waitforbuttonpress()
#     for ele in sol:
#         print ele
#     print np.mean(np.array(time_array))
#
#


























