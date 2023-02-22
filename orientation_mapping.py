import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance, distance_matrix
from scipy.interpolate import interp2d
from scipy.spatial.distance import euclidean
from skimage.feature import peak_local_max


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

def get_angles(vor,ideal_vectors,angle_ideal, rotation_angle = 0):
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
    vector_nor=vector_sum/number
    #loop, sort in anti-clockwise, match, label
    angles=[]
    RMSDs=[]
    for x in tqdm(range(vor.npoints)):
        vectors=vor.vertices[vor.regions[vor.point_region[x]]]-vor.points[x]
        if vectors.shape[0]!=6:
            angles.append(0);continue;
        RMSD, angle = matching(vectors,ideal_vectors,angle_ideal,3)
        angles.append(angle)
        RMSDs.append(RMSD)
    for i in range(len(angles)):
        while angles[i] < 0:
            angles[i] += np.pi
        while angles[i] > np.pi:
            angles[i] -= np.pi
        angles[i] += rotation_angle

    return angles