import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.io import imread, imshow
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy.ndimage import gaussian_filter
import cv2
import os
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance, distance_matrix
from cProfile import Profile
import scipy.ndimage as nd
from scipy.signal import find_peaks,peak_prominences
from scipy.optimize import curve_fit
from multiprocessing import Process
from scipy import fftpack
from skimage import measure
from scipy.interpolate import interp2d
from scipy.spatial.distance import euclidean
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
import networkx as nx
from scipy import spatial
from copy import deepcopy

def get_graph(points, bond_thres = False, node_info=None):
    vor = Voronoi(points)
    G = nx.Graph()
    G.add_nodes_from(range(len(vor.points)))
    G.add_edges_from(vor.ridge_points)
    for i in G.nodes:
        G.nodes[i]['loc'] = points[i]
        
        #### remove edge too long
    ridge_length = []
    for x,y in vor.ridge_points:
        length = np.linalg.norm(vor.points[x] - vor.points[y])
        ridge_length.append(length)

    ridge_length = np.array(ridge_length)
    ridge_length_med = np.median(ridge_length)
    if not bond_thres:
        bond_thres = ridge_length_med*1.5

    edges_to_remove = []
    for i in range(len(ridge_length)):
        if ridge_length[i] > bond_thres:                 # hyperparameter
            edges_to_remove.append(vor.ridge_points[i])

    edges_to_remove = np.array(edges_to_remove)

    G.remove_edges_from(edges_to_remove)

        
    if isinstance(node_info, np.ndarray):
        if node_info.shape[0] != points.shape[0]: print("node_info length not equal to number of points")
        for i in range(points.shape[0]):
            G.nodes[i]['index'] = node_info[i]
            
    # remove nodes with two or less edges
    while True:
        nodes_to_remove = []
        for i in G.nodes:
            if len(G.edges(i))<=2:
                nodes_to_remove.append(i)    
        if nodes_to_remove == []: break
        G.remove_nodes_from(nodes_to_remove)

    return G

def find_med_nn_dist(obj_cors):
    vor = Voronoi(obj_cors)
    dist_list = np.zeros(vor.ridge_points.shape[0])
    for i in range(vor.ridge_points.shape[0]):
        x, y = vor.ridge_points[i,:]
        dist_list[i] = np.linalg.norm(vor.points[x] - vor.points[y])
    return np.median(dist_list)


def find_nestest_point(obj_cor, candi_cors, thres):
    nearest_dist = np.inf
    tree = spatial.cKDTree(candi_cors)
    indices = tree.query_ball_point(obj_cor, thres)
    for indice in indices:
        dist = np.linalg.norm(candi_cors[indice] - obj_cor)
        if dist < nearest_dist: nearest_dist = dist
    return nearest_dist


def find_nestest_dists(obj_cors, candi_cors): 
    thres = find_med_nn_dist(obj_cors)
    dist_list = np.zeros(obj_cors.shape[0])
    for i, obj_cor in enumerate(tqdm(obj_cors)):
        dist_list[i] = find_nestest_point(obj_cor, candi_cors, thres)
    return dist_list

def find_nestest_dists(obj_cors, candi_cors):   #fast version
    tree = spatial.cKDTree(candi_cors)
    distances, ndx = tree.query(obj_cors)
    return distances

def get_displacement_field(cur_cors, pre_cors, upper_lim):
    tree = spatial.cKDTree(pre_cors)
    distances, ndx = tree.query(cur_cors)
    vectors = np.zeros([cur_cors.shape[0], 2])
    for i in range(cur_cors.shape[0]):
        vector = cur_cors[i] - pre_cors[ndx[i]]
        if np.linalg.norm(vector) > upper_lim: continue;
        vectors[i] = vector
    return vectors

def get_dist_list(grain_atoms, coordinates_all):
    dist_all = np.zeros([coordinates_all.shape[0], len(grain_atoms)])
    for j, coordinates in enumerate(grain_atoms):
        dist_all[:,j] = find_nestest_dists(coordinates_all, coordinates)
    return dist_all

def label_atoms_by_dist(dist_all, thres):
    label_list = np.zeros(dist_all.shape[0])
    ## if dist lower than thres for both, label as 0
    ## if dist lower than thres for one, label as 1/2
    ## if dist larger than thres for both, label as 3
    for i in range(dist_all.shape[0]):
        if dist_all[i,:].min()>thres:
            label_list[i] = 0
        else:
            label_list[i] = dist_all[i,:].argmin()+1
#         if dist_all[i,0]<thres and dist_all[i,1]<thres:
#             label_list[i] = 0
#         elif dist_all[i,0] < thres:
#             label_list[i] = 1
#         elif dist_all[i,1] < thres:
#             label_list[i] = 2
#         else:
#             label_list[i] = 3
    return label_list

def get_neighbor(i,vor):
    return np.concatenate([vor.ridge_points[vor.ridge_points[:,0]==i][:,1],vor.ridge_points[vor.ridge_points[:,1]==i][:,0]])

def get_connected(points, dist_list, thres_dist, connectivity, thres_atom_number, median_interatomic_dist): # break one bond connection, return connected components
    ## suit only for high quality images
    bool_list = dist_list < thres_dist
    new_dist_list = np.ones_like(dist_list)*np.inf#*np.inf
    components_return = []
    for i in range(bool_list.shape[1]):
        points_sel = points[bool_list[:,i]]
        G = get_graph(points_sel, median_interatomic_dist*1.5, node_info = np.where(bool_list[:,i])[0])
        components = nx.k_edge_components(G, k=connectivity)
        for j in components:
            if len(j)>thres_atom_number: 
                print(i, len(j)); 
                components_return.append(points_sel[list(j)]);
                for atom in j:  # get new distance list, filtered unconnected components
                    new_dist_list[G.nodes[atom]['index'], i] = dist_list[G.nodes[atom]['index'], i]
    return components_return, new_dist_list

def get_graph_for_cut(points, dist_list, capacity):
    vor = Voronoi(points)
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(len(vor.points)))
    G.add_edges_from(vor.ridge_points, capacity = capacity)
    G.add_edges_from(vor.ridge_points[:,::-1], capacity = capacity)
    #### remove edge too long
    ridge_length = []
    for x,y in vor.ridge_points:
        length = np.linalg.norm(vor.points[x] - vor.points[y])
        ridge_length.append(length)

    ridge_length = np.array(ridge_length)
    ridge_length_med = np.median(ridge_length)
    edges_to_remove = []
    for i in range(len(ridge_length)):
        if ridge_length[i] > ridge_length_med*1.2:                 # hyperparameter
            edges_to_remove.append(vor.ridge_points[i])

    edges_to_remove = np.array(edges_to_remove)

    G.remove_edges_from(edges_to_remove)
    
    for i,c in enumerate(dist_list):
#         c[0], c[1] = min(1/c[0],100), min(100,1/c[1])
#         c[0], c[1] = min(c[0],100), min(100,c[1])
        cap_1, cap_2 = min(c[0],10), min(10,c[1])
        cap_1, cap_2 = np.exp(cap_1), np.exp(cap_2)
        G.add_edge('s', i, capacity=cap_1)
        G.add_edge(i, 't', capacity=cap_2)
    return G

def clean_labels(points, labels): # remove isolated atomsby counting neighbors
    vor = Voronoi(points)
    for i in range(len(points)):
        neighbors = get_neighbor(i,vor)
        count = 0
        for j in neighbors:
            if labels[j] == labels[i]:
                count += 1
        if count <= 1:
            labels[i] = 1-labels[i]
    return labels
    
def clean_labels(points, labels): # remove isolated atoms by connected_component
    vor = Voronoi(coordinates_all)
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(vor.points)))
    G.add_edges_from(vor.ridge_points)
    for i in G.nodes:
        G.nodes[i]['loc'] = coordinates_all[i]
        G.nodes[i]['label'] = labels[i]
    #### remove edge too long
    ridge_length = []
    for x,y in vor.ridge_points:
        length = np.linalg.norm(vor.points[x] - vor.points[y])
        ridge_length.append(length)

    ridge_length = np.array(ridge_length)
    ridge_length_med = np.median(ridge_length)
    edges_to_remove = []
    for i in range(len(ridge_length)):
        if ridge_length[i] > ridge_length_med*1.5:                 # hyperparameter
            edges_to_remove.append(vor.ridge_points[i])
    #### remove edge between two type of labels
    for i,j in G.edges:
        if G.nodes[i]['label'] != G.nodes[j]['label']:
            edges_to_remove.append(np.array([i,j]))
    edges_to_remove = np.array(edges_to_remove)
    G.remove_edges_from(edges_to_remove)
    #### remove low connected_component
    if len(list(nx.connected_components(G)))>2:
        for connected_components in list(nx.connected_components(G))[2:]:
            for component in connected_components:
                labels[component] = 1-labels[component]
    return labels

def reassign_orphan_atoms(points, labels, dist_list):
    vor = Voronoi(points)
    labels_ = deepcopy(labels)
    while (labels_ == 0).sum()!=0:
        print((labels_ == 0).sum())
        for atom_id in np.where(labels_ == 0)[0]:
            neighbors = get_neighbor(atom_id,vor)
            neighbor_labels = np.unique(labels_[neighbors])
            neighbor_labels = neighbor_labels[neighbor_labels!=0].astype(int)
            if len(neighbor_labels) == 1:
                labels_[atom_id] = neighbor_labels[0]
            elif len(neighbor_labels) > 1:
                labels_[atom_id] = dist_list[atom_id, :].argmin()+1
    return labels_

def if_GB_atom(vor, atom_id, labels):
    neighbors = get_neighbor(atom_id,vor)
    label = np.unique(labels[neighbors])
    if len(label) == 1: return True
    else: return False
    

def optimize_GBs(points, labels_, dist_list):
    labels = deepcopy(labels_)
    vor = Voronoi(points)
    boundary_atoms = []
    ridge_length = []
    for x,y in vor.ridge_points:
        length = np.linalg.norm(vor.points[x] - vor.points[y])
        ridge_length.append(length)
    ridge_length = np.array(ridge_length)
    ridge_length_med = np.median(ridge_length)
    
    for i,j in vor.ridge_points:
        if labels[i] != labels[j] and np.linalg.norm(vor.points[i]-vor.points[j])<1.3*ridge_length_med:
            boundary_atoms.append(i)
            boundary_atoms.append(j)
    boundary_atoms = np.unique(boundary_atoms)
    # optimization
    if_moved = 1
    while if_moved:
        new_boundary_atoms = []
#     print(boundary_atoms)
        for GB_atom in boundary_atoms:
            neighbors = get_neighbor(GB_atom,vor)
            neighbor_labels = np.unique(labels[neighbors]).astype(int)
            if dist_list[GB_atom, labels[GB_atom].astype(int)-1] != dist_list[GB_atom, neighbor_labels-1].min():
#                 print(labels[GB_atom].astype(int)-1,dist_list[GB_atom, :].argmin())
#                 print(dist_list[GB_atom, labels[GB_atom].astype(int)-1], dist_list[GB_atom, neighbor_labels-1].min())
#                 print('moving', GB_atom)
                labels[GB_atom] = np.where(dist_list[GB_atom,:] == dist_list[GB_atom, neighbor_labels-1].min())[0][-1]+1
                for neighbor in neighbors:
                    if if_GB_atom(vor, neighbor, labels):
                        new_boundary_atoms.append(neighbor)
        boundary_atoms = np.unique(new_boundary_atoms)
        if len(boundary_atoms)==0: if_moved=0
    return labels

def coh_GB_atoms(points, labels, dist_list, coh_thres):
    vor = Voronoi(points)
    boundary_atoms = []
    ridge_length = []
    for x,y in vor.ridge_points:
        length = np.linalg.norm(vor.points[x] - vor.points[y])
        ridge_length.append(length)
    ridge_length = np.array(ridge_length)
    ridge_length_med = np.median(ridge_length)
    
    for i,j in vor.ridge_points:
        if labels[i] != labels[j] and np.linalg.norm(vor.points[i]-vor.points[j])<1.3*ridge_length_med:
            boundary_atoms.append(i)
            boundary_atoms.append(j)
    boundary_atoms = np.unique(boundary_atoms)
    coh_atoms = []
    for GB_atom in boundary_atoms:
        neighbors = get_neighbor(GB_atom,vor)
        neighbor_labels = np.unique(labels[neighbors]).astype(int)
        GB_atom_label_bool = dist_list[GB_atom, neighbor_labels] < coh_thres
        if GB_atom_label_bool.sum()>1:
            thres_labels = np.where(dist_list[GB_atom, neighbor_labels] < coh_thres)[0]
            atom_idx = neighbor_labels[:2]
#             print(neighbor_labels)
            coh_atoms.append([GB_atom, atom_idx[0],atom_idx[1]])
    return np.array(coh_atoms)

def potts(G, labels, weight=1):
    costs = dict()
    for s in labels:
        for t in labels:
            costs[s,t] = weight * int(s != t)
    return costs

#Compute alpha-expansion optimization graph
def expansion_graph(G, f, alpha, data_costs, edge_costs):
    G_a = nx.Graph()
    
    #Add nodes
    G_a.add_nodes_from([e for e in G.nodes()])
    G_a.add_node( 'alpha' ) 
    G_a.add_node( 'nonalpha' )
    
    for e in G.edges():
        u,v = e 
        if f[u] == f[v]:
            G_a.add_node( e )
    
    #Add edges with unary costs
    for v in G.nodes():
        G_a.add_edge( v,'alpha', weight=data_costs[v][alpha] )
        if f[v] == alpha:
            G_a.add_edge( v,'nonalpha', weight=float('inf') )
        else:
            G_a.add_edge( v,'nonalpha', weight=data_costs[v][f[v]] )
    
    #Add edges with pairwise costs
    for e in G.edges():
        u,v = e
        if f[u] == f[v]:
            G_a.add_edge( u,v, weight=edge_costs[alpha, f[u]] )
        else:
            G_a.add_edge( u, e, weight=edge_costs[alpha, f[u]] )
            G_a.add_edge( v, e, weight=edge_costs[alpha, f[v]] )
            G_a.add_edge( e, 'nonalpha', weight=edge_costs[f[u],f[v]] )
    
    return G_a


def alpha_expansion(G, f, alpha, data_costs, edge_costs):
    G_a = expansion_graph(G,f,alpha, data_costs, edge_costs)
    cutvalue, partition = nx.minimum_cut(G_a, 'alpha', 'nonalpha', capacity='weight')
    S,T = partition
    for v in G.nodes():
        if v in T:
            f[v] = alpha
    return f



