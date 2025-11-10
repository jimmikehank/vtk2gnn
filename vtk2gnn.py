import os
import pyvista as pv
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import LayerNorm
from tqdm import tqdm

def sort_vtk(case='./'):
    # Function for sorting contents of vtk files
    # case variable should be an OpenFOAM case directory
    # Returns list of paths to vtk files in order by timestep
    import numpy as np
    target = case + '/VTK/'
    full = [i for i in os.listdir(target) if os.path.isfile(target + i)]
    iters = np.array([int(item.split('_')[-1].split('.')[0]) for item in full])
    order = np.argsort(iters)
    full_sorted = [target + full[i] for i in order]
    return full_sorted

class Graph():
    def __init__(self,mesh_data,xbounds = [-1,1], ybounds = [-1,1]):
        self.inds = self.filter_cell_area(mesh_data,xbounds,ybounds)
        filtered_mesh = mesh_data.extract_cells(self.inds)
        self.init_graph(filtered_mesh)           
        self.edge_index = torch.tensor(self.neighbors).t().contiguous().long()
        
    def filter_cell_area(self,mesh_data,xbounds,ybounds):
        # Function for filtering of data by area. 
        # Takes pyvista mesh data, [left,right] and [bottom,top].
        # Returns indices of cells within boundary region
        pts = mesh_data.cell_centers().points
        m,n = np.shape(pts)
        west = xbounds[0]
        east = xbounds[1]
        north = ybounds[1]
        south = ybounds[0]
        ptinds = [i for i in range(m) if pts[i,0] > west and pts[i,0] < east and pts[i,1] > south and pts[i,1] < north]
        return ptinds

    def get_neighbors(self,mesh_data):
        # Function for creating neighbor list.
        # Returns list of all face shared neighbor cells.
        neighbors = []
        for i in range(mesh_data.n_cells):
            local = mesh_data.cell_neighbors(i,'faces')
            for j in local:
                neighbors.append([i,j])
        return neighbors

    def init_graph(self,mesh_data):
        self.neighbors = self.get_neighbors(mesh_data)
        self.points = torch.from_numpy(mesh_data.cell_centers().points[:,:2])

    def retrieve_data(self, target):
        # Function for extracting pressure and velocity data.
        # Loads a single snapshot at target, returns U, p fields in filtered zone.        
        mesh_data = pv.read(target)
        u = torch.from_numpy(mesh_data.cell_centers()['U'][self.inds])
        p = torch.from_numpy(mesh_data.cell_centers()['p'][self.inds]).unsqueeze(-1)
        return torch.cat((u[:,:2],p),dim=-1)

    def gen_data_single(self, target):
        nodes = self.retrieve_data(target)
        data = Data(x=nodes,edge_index=self.edge_index)
        return data

class GCN(torch.nn.Module):
    # Generic GCN model for testing
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.norm = LayerNorm(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.norm(x)
        return x

if __name__ == '__main__':
    target_directory = '/home/james/Documents/research/data/lr500/' # Your local /path/to/foam/case
    targets = sort_vtk(target_directory) # Sort vtk files in foam case
    init_data = pv.read(targets[0]) # Dataset for graph initialization
    xbounds = [-1.00, 0.08] # X - bounds for FO case
    ybounds = [-0.10, 0.10] # Y - bounds for FO case
    graph = Graph(init_data,xbounds,ybounds) # Initialize graph to generate points and neighbor data 

    nsteps = 400 # Number of snapshots to use
    start = 100 # First snapshot for dataset
    data = []

    # This for loop generates a list of data to step through
    for i in range(start,start+nsteps):
        data.append(graph.gen_data_single(targets[i]))