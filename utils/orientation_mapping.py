import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance, distance_matrix
from scipy.interpolate import interp2d
from scipy.spatial.distance import euclidean
from skimage.feature import peak_local_max
import cv2
from scipy import fftpack
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import networkx as nx

mappings= [
    [0,1,2,3,4,5],
    [1,2,3,4,5,0],
    [2,3,4,5,0,1],
    [3,4,5,0,1,2],
    [4,5,0,1,2,3],
    [5,0,1,2,3,4],
]

def get_ideal_factor_for_fcc(orientation):
    lattice=[]
    if orientation == '111':
        for i in range(10):
            for j in range(10):
                lattice.append([i+j/2,j*np.sqrt(3)/2])
    elif orientation == '110':
        for i in range(10):
            for j in range(10):
                lattice.append([i*np.sqrt(6)/4+1/3*j*np.sqrt(6)/4,j*np.sqrt(3)/3])
    else:
        raise Exception("Sorry this orientation hasn't been implemented, pls define it in the function")
    lattice=np.array(lattice)
    vor = Voronoi(lattice)
    x=15
    SUM=0
    num=0
    ver=vor.vertices[vor.regions[vor.point_region[x]]]
    for i in ver:
        SUM=SUM+(distance.euclidean(i,lattice[x]))
        num=num+1
    AVE=SUM/num
    ver=ver/AVE
    lattice=lattice/AVE
    displacement=[]
    for i in ver:
        displacement.append(i-lattice[x])
    displacement=np.array(displacement)
    angle=[]
    for j in displacement:
        angle.append(np.arctan2(j[1],j[0]))
    angle=np.array(angle)
    displacement = displacement[np.argsort(angle)]
    ideal_vectors=displacement
    angle_ideal = angle[np.argsort(angle)]
    return ideal_vectors, angle_ideal


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def cal_RMSD(data_vectors,ideal_vectors, mapping, num=6):
    RMSD=0
    for i in range(num):
        RMSD=RMSD+rmse(data_vectors[mapping[i]],ideal_vectors[i])
    return RMSD/num
    
def matching(vectors,ideal_vectors,angle_ideal,num=3):    #return RMSD_min and angle
    RMSD_min=100
    angle_min=0
    angle=[]
    for j in vectors:                                           #sort vectors and angles
        angle.append(np.arctan2(j[1],j[0]))
    angle=np.array(angle)
    vectors = vectors[np.argsort(angle)]
    angle = angle[np.argsort(angle)]
    misorientation = angle - angle_ideal[0]
    
    for i in range(num):
        mapping = mappings[i]
        theta=misorientation[i]
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.transpose(((c, -s), (s, c)))
        ideal_rotate = np.dot(ideal_vectors,rotation_matrix)
        RMSD=cal_RMSD(vectors,ideal_rotate,mapping)
        if RMSD<RMSD_min:RMSD_min=RMSD;angle_min=theta;
    return RMSD_min,angle_min

def find_median_interatomic_dist(vor):
    ridge_length = []
    for x,y in vor.ridge_points:
        length = np.linalg.norm(vor.points[x] - vor.points[y])
        ridge_length.append(length)
    ridge_length = np.array(ridge_length)
    median_interatomic_dist = np.median(ridge_length)
    return median_interatomic_dist

def get_angles(vor,ideal_vectors,angle_ideal, rotation_angle = 0, normalization = True, return_error = False):
    vector_sum=0
    number=0
    median_interatomic_dist = find_median_interatomic_dist(vor)
    for x in range(vor.npoints):
        ver=vor.regions[vor.point_region[x]]
        if len(ver)==6:
            for i in ver:
                dist=distance.euclidean(vor.points[x],vor.vertices[i])
                if dist<2*median_interatomic_dist:  
                    vector_sum=vector_sum + dist
                    number=number+1
    if normalization:
        vector_nor=vector_sum/number
    #loop, sort in anti-clockwise, match, label
    angles=[]
    RMSDs=[]
    for x in tqdm(range(vor.npoints)):
        vectors=vor.vertices[vor.regions[vor.point_region[x]]]-vor.points[x]
        if vectors.shape[0]!=6:
            angles.append(0);RMSDs.append(1e5);continue;
        RMSD, angle = matching(vectors,ideal_vectors,angle_ideal,3)
        angles.append(angle)
        RMSDs.append(RMSD)
    for i in range(len(angles)):
        while angles[i] < 0:
            angles[i] += np.pi
        while angles[i] > np.pi:
            angles[i] -= np.pi
        angles[i] += rotation_angle
    if return_error:
        return angles, RMSDs
    return angles


###################################################
""" template generation"""
###################################################
def select_region_for_FFT(img):
    roi = cv2.selectROI('roi',img, False, False)
    cv2.destroyAllWindows()
    left, right, bottom, top = roi[1], roi[1]+roi[3], roi[0], roi[0]+roi[2]
    img_new = deepcopy(img)
    img_new = cv2.rectangle(img_new, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (255, 0, 0), 2)
    plt.figure()
    plt.imshow(img_new)
    return img[left:right, bottom:top]

def manual_select_diff_spots(fft_img, Peak_end = False):
    if Peak_end:
        left, right, bottom, top = fft_img.shape[1]//2 - Peak_end, fft_img.shape[1]//2 + Peak_end, fft_img.shape[0]//2 - Peak_end, fft_img.shape[0]//2 + Peak_end
    else:
        left, right, bottom, top = 0, fft_img.shape[1], 0, fft_img.shape[0]
    ROIs = cv2.selectROIs('rois',fft_img[left:right, bottom:top], False, False)
    cv2.destroyAllWindows()
    print(left, right, bottom, top)
    img_new = deepcopy(fft_img)
    plt.figure()
    for roi in ROIs:
        roi[0] = roi[0]+left
        roi[1] = roi[1]+bottom
        img_new = cv2.rectangle(img_new, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (255, 0, 0), 2)
    plt.imshow(img_new[left:right, bottom:top])
    return ROIs

def get_smooth_FFT(img, sigma = 5):
    F1 = fftpack.fft2((img).astype(float))
    F2 = fftpack.fftshift(F1)
    im1 = (20*np.log10( 0.1 + F2)).astype(int)
    im1 = gaussian_filter(im1,sigma) 
    im1 = rescale_intensity(im1, out_range=(0,255)).astype(np.uint8)
    return im1

def get_maximum_from_POIs(img, ROIs):
    locs = []
    fft_img = get_smooth_FFT(img, sigma = 0)
    for roi in ROIs:
        left, right, bottom, top = roi[1], roi[1]+roi[3], roi[0], roi[0]+roi[2]
        loc_fft = fft_img[left:right, bottom:top]
        max_loc = np.unravel_index(loc_fft.argmax(), loc_fft.shape)
        locs.append([max_loc[0]+left, max_loc[1]+bottom])
    return locs

def template_map(img, Peak_end = False):
    img = select_region_for_FFT(img)
    fft_img = get_smooth_FFT(img, sigma = 0)
    ROIs = manual_select_diff_spots(fft_img, Peak_end)
    max_locs = get_maximum_from_POIs(img, ROIs)
    mask = np.zeros_like(fft_img)
    for i in max_locs:
        mask[i[0],i[1]] = 1
    im1 = fftpack.ifft2(fftpack.ifftshift(mask*fft_img)).real
    print('spot of interests = ', max_locs)
    return im1

def get_template_factors(img_template):
    lattice = peak_local_max(img_template, min_distance=5,exclude_border=5)
    vor = Voronoi(lattice)
    x = 0    
    ver=vor.vertices[vor.regions[vor.point_region[x]]]
    while len(ver)!=6 and x<lattice.shape[0]: 
        x += 1
        ver=vor.vertices[vor.regions[vor.point_region[x]]]
    displacement=[]
    for i in ver:
        displacement.append(i-vor.points[x])
    displacement=np.array(displacement)
    angle=[]
    for j in displacement:
        angle.append(np.arctan2(j[1],j[0]))
    angle=np.array(angle)
    displacement = displacement[np.argsort(angle)]
    ideal_vectors=displacement
    angle_ideal = angle[np.argsort(angle)]
    return ideal_vectors, angle_ideal


#######################################################
""" Lattice relaxation"""
#######################################################
def get_graph_relaxation(points, bond_thres = False, node_info=None):
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
    # define k and l0 for every edge
    for edge in G.edges:
        G[edge[0]][edge[1]]['k'] = 1
        G[edge[0]][edge[1]]['l0'] = 0 #ridge_length.mean(), to be adjusted
    return G

def bond_force(G, node):
    net_force = np.zeros(2)
    for n1, n2, data in G.edges(node, data=True):
        # Calculate the force exerted by the spring using Hooke's law
        vector = G.nodes[n2]['loc'] - G.nodes[n1]['loc']
        spring_force = data['k'] * (vector - data['l0']*vector/np.linalg.norm(vector))
        # Add the spring force to the net force
        net_force += spring_force
    return net_force


def image_force(image, loc):
    net_force = np.zeros(2)
    sobel_kernel_x = -1*np.array([[1,0,-1],[2,0,-2],[1,0,-1]]).T
    sobel_kernel_y = -1*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    local_area = image[loc[0]-1:loc[0]+2,loc[1]-1:loc[1]+2]
    net_force[0] = (sobel_kernel_x*local_area).sum()
    net_force[1] = (sobel_kernel_y*local_area).sum()
    return net_force


def lattice_relax(G, image, epoch, beta, step, max_bond_force):
    disp = []
    for i in G.nodes:
        G.nodes[i]['bond_force'] = bond_force(G,i)
    for i in G.nodes:
        loc = G.nodes[i]['loc'].astype(int)
        G.nodes[i]['image_force'] = image_force(image,loc)/10
    for _ in tqdm(range(epoch)):
        total_disp = np.zeros(2)
        for i in G.nodes:
            loc = G.nodes[i]['loc'].astype(int)
            G.nodes[i]['bond_force'] = bond_force(G,i) if np.linalg.norm(bond_force(G,i)) < max_bond_force else max_bond_force*bond_force(G,i)/np.linalg.norm(bond_force(G,i))
            G.nodes[i]['image_force'] = image_force(image,loc)
            G.nodes[i]['net_force'] = beta*G.nodes[i]['bond_force'] + G.nodes[i]['image_force']/20
        for i in G.nodes:
            if len(list(G.neighbors(i))) != 6:
                continue;
            G.nodes[i]['loc'] = G.nodes[i]['loc'] + G.nodes[i]['net_force']*step
            total_disp = total_disp + np.linalg.norm(G.nodes[i]['net_force']*step)
        disp.append(total_disp)
    return disp