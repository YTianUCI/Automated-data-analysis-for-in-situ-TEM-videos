import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance_matrix
from scipy.interpolate import interp2d
from scipy.spatial.distance import euclidean
from skimage.feature import peak_local_max
import cv2
from scipy.ndimage.measurements import center_of_mass
from scipy import fftpack
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import networkx as nx
import tifffile
import math
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_tiff_frame(filepath, frame_num=0, num_frames=1):
    """
    Reads and averages a sequence of frames from a TIFF stack.

    Args:
        filepath (str): Path to the TIFF stack file.
        frame_num (int): Index of the first frame to read (default: 0).
        num_frames (int): Number of consecutive frames to average (default: 1).

    Returns:
        numpy.ndarray: The averaged frame as a NumPy array.
    """
    with tifffile.TiffFile(filepath) as tif:
        frames = tif.asarray(key=slice(frame_num, frame_num+num_frames)).astype(np.float32)
        if num_frames>1:
            avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        else:
            avg_frame = frames.astype(np.uint8)
        return avg_frame
    
def read_image(path, frame_number):
    """
    Reads and averages a frames from a video.

    Args:
        filepath (str): Path to the video file.
        frame (int): Index of the frame to read.

    Returns:
        numpy.ndarray: The averaged frame as a NumPy array.
        int: The number of frames in the video.
    """
    cap = cv2.VideoCapture(path)
    frames = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT) ))
    width=int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_WIDTH)))-1
    height=int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_HEIGHT)))-1
    cap.set(1, frame_number-1)
    res, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, frames

mappings= [
    [0,1,2,3,4,5],
    [1,2,3,4,5,0],
    [2,3,4,5,0,1],
    [3,4,5,0,1,2],
    [4,5,0,1,2,3],
    [5,0,1,2,3,4],
]

def mappings(x):
    """
    Given an integer x, returns a 2D numpy array of shape (x, x) representing the mappings.
    
    Args:
    x: An integer representing the size of the array.
    
    Returns:
    A 2D numpy array of shape (x, x) representing the mappings.
    """
    result = np.zeros([x,x],dtype=int)
    for i in range(x):
        result[i] = np.roll(np.arange(x),-i)
    return result
    
def sort_angle(displacement):
    angle=[]
    for j in displacement:
        angle.append(np.arctan2(j[1],j[0]))
    angle=np.array(angle)
    displacement = displacement[np.argsort(angle)]
    return displacement, angle[np.argsort(angle)]


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
        SUM=SUM+(euclidean(i,lattice[x]))
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
    """
    Compute the root mean square error between two arrays.

    Parameters
    ----------
    predictions : numpy array
        The predicted values.
    targets : numpy array
        The true values.

    Returns
    -------
    rmse : float
        The root mean square error between the two arrays.

    """
    return np.sqrt(((predictions - targets) ** 2).mean())

def cal_RMSD(data_vectors,ideal_vectors, mapping, num=6):
    """
    Calculate the root mean square deviation between two sets of vectors.

    Parameters
    ----------
    data_vectors : numpy array
        The set of data vectors.
    ideal_vectors : numpy array
        The set of ideal vectors.
    mapping : list
        A list of mappings between the data vectors and the ideal vectors.
    num : int, optional
        The number of vectors to consider in the calculation. Default is 6.

    Returns
    -------
    RMSD : float
        The root mean square deviation between the two sets of vectors.

    """
    RMSD=0
    for i in range(num):
        RMSD=RMSD+rmse(data_vectors[mapping[i]],ideal_vectors[i])
    return RMSD/num

def image_interpolation(image, factor):

    new_size = (image.shape[0] * factor, image.shape[1] * factor)

    # Create the interpolation function
    interp_func = interp2d(np.arange(image.shape[0]), np.arange(image.shape[1]), image, kind='cubic')

    # Evaluate the interpolation function at the new grid
    new_img_array = interp_func(np.linspace(0, image.shape[0] - 1, new_size[0]), np.linspace(0, image.shape[1] - 1, new_size[1]))

    # Convert the array back to an image and save it
    image = np.uint8(new_img_array)

    return image

def subpixel_refine(loc, r, image):
    """
    Refine the location of an atomic site to subpixel accuracy using image processing.

    less
    Copy code
    Parameters
    ----------
    loc : tuple
        The initial location of the atomic site.
    r : int
        The radius of the circle to use for the mask.
    image : numpy array
        The input image.

    Returns
    -------
    x, y : float
        The refined location of the atomic site.

    """
    x, y = loc
    left, right, top, bottom = int(max(x-r, 0)), int(min(x+r+1, image.shape[0])), int(max(y-r, 0)), int(min(y+r+1, image.shape[1]))
    img = image[left:right,top:bottom]
    mask = np.zeros(img.shape,dtype=np.uint8)
    center = (r,r)
    cv2.circle(mask, center, r, 1, -1)
    masked_img = img*mask
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.scatter(center[1], center[0],s=1.5, color = 'r')
    x, y = center_of_mass(masked_img)
    # plt.scatter(y, x,s=1.5, color = 'b')
    x+=left
    y+=top
    return x, y

def refine_atom_locs(image: np.ndarray, locs: np.ndarray, r: int) -> np.ndarray:
    """
    Refine the locations of atoms in an image to subpixel accuracy.

    Parameters
    ----------
    image : numpy array
        The input image.
    locs : numpy array
        An array of atom locations.
    r : int
        The radius around each atom to refine.

    Returns
    -------
    new_locs : numpy array
        An array of refined atom locations.

    """
    # Create new array to store subpixel refined locations
    new_locs = np.zeros_like(locs,dtype=float)
    # Loop over locations
    for i, loc in enumerate(locs):
        # Refine location
        new_locs[i] = subpixel_refine(loc, r, image)
    # Return refined locations
    return new_locs

def decompose(A):
    """
    Decompose a matrix into a rotation matrix and a deformation matrix.

    Parameters
    ----------
    A : numpy array
        The input matrix.

    Returns
    -------
    R : numpy array
        The rotation matrix.
    D : numpy array
        The deformation matrix.

    """
    U, s, V = np.linalg.svd(A)
    # Extract the rotation matrix and deformation matrix
    R = np.dot(U, V)
    D = np.dot(V.T, np.dot(np.diag(s), V))
    return R, D

def optimize_M(x0, src, dst, num_of_neighbors=6):
    """
    Optimize the affine transformation matrix M using a set of source and destination points.

    Parameters
    ----------
    x0 : numpy array
        Initial guess for the affine transformation parameters.
    src : numpy array
        An array of source points.
    dst : numpy array
        An array of destination points.
    num_of_neighbors : int, optional
        The number of neighboring points to consider. Default is 6.

    Returns
    -------
    M : numpy array
        The optimized affine transformation matrix.

    """
    # Construct the matrix A
    A = np.zeros((num_of_neighbors*2, 6))
    for i in range(num_of_neighbors):
        x, y = src[i]
        u, v = dst[i]
        A[2*i] = [x, y, 1, 0, 0, 0]
        A[2*i+1] = [0, 0, 0, x, y, 1]
    
    # Construct the matrix b
    b = dst.reshape(num_of_neighbors*2, 1)
    
    # Solve the linear system Ax = b
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Construct the affine transformation matrix
    M = np.vstack((x.reshape(2, 3), [0, 0, 1]))

    return M[:2,:2]

def matching(vectors,ideal_vectors,angle_ideal,num=3):    #return RMSD_min and angle
    """
    Find the optimal rotation and RMSD between two sets of vectors.

    Parameters
    ----------
    vectors : numpy array
        The set of input vectors.
    ideal_vectors : numpy array
        The set of ideal vectors.
    angle_ideal : float
        The ideal angle of rotation.
    num : int, optional
        The number of mappings to consider. Default is 3.

    Returns
    -------
    RMSD_min : float
        The minimum root mean square deviation between the two sets of vectors.
    angle_min : float
    The angle of rotation that gives the minimum RMSD.

    """
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

def cluster_and_average(points, num_clusters=4):
    """
    Clusters a group of points into 4 groups using k-means algorithm from the sklearn package.
    Calculates the average point center of each cluster.
    
    Arguments:
    points -- a numpy array of shape (n, 2) representing n points in 2-dimensional space
    
    Returns:
    labels -- a numpy array of shape (n,) containing the cluster labels for each point
    centers -- a numpy array of shape (4, 2) containing the average point center of each cluster
    """
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(points)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return labels, centers

def get_template_factors(img_template,num_of_neighbors,neighbor_distance,given_point=None,plot=False):
    """
    Given an image template and the number of neighbors, returns the ideal vectors and angles.
    
    Args:
    img_template: An image template.
    num_of_neighbors: An integer representing the number of neighbors.
    
    Returns:
    A tuple containing the ideal vectors and angles.
    """
    coordinates_all = peak_local_max(img_template, min_distance=int(neighbor_distance/2),exclude_border=5)
    coordinates_all = refine_atom_locs(img_template, coordinates_all, int(median_template_dist/1.5))
    lattice = np.zeros((len(lattice),7))
    st.afit.atom_positions.refine_atoms_numba(img_template, coordinates_all,lattice,median_interatomic_dist//2)
    if plot:
        plt.figure()
        plt.imshow(img_template)
    plt.scatter(lattice[:, 1], lattice[:, 0],s=1.5, color = 'r')
    vor = Voronoi(lattice)
    template_vectors = []
    if given_point:
        displacement = get_neighbor_vectors(vor,given_point,num_of_neighbors)
        ideal_vectors, angle_ideal = sort_angle(displacement)
        return ideal_vectors, angle_ideal
    for x in range(vor.npoints):
        displacement = get_neighbor_vectors(vor,x,num_of_neighbors)
        if (np.linalg.norm(displacement,axis=1) > neighbor_distance*1.5).sum() == 0:
            template_vectors.append(displacement)
    template_vectors = np.vstack(template_vectors)
    _, centers = cluster_and_average(template_vectors.reshape(-1,2),num_clusters=num_of_neighbors)
    ideal_vectors, angle_ideal = sort_angle(centers)
    return ideal_vectors, angle_ideal


def matching_all(vectors,ideal_vectors,angle_ideal,num=3,num_of_neighbors=6):    #revised one, to give distortion fitting results
    """
    Given vectors, ideal vectors, and an ideal angle, returns the RMSD error, angle, and coefficients of a polynomial fit to the distortion.
    
    Args:
    vectors: A numpy array of shape (num_of_neighbors, 2) representing the vectors.
    ideal_vectors: A numpy array of shape (6, 2) representing the ideal vectors.
    angle_ideal: A float representing the ideal angle between the vectors.
    num: An integer representing the number of times to run the function. Default is 3.
    num_of_neighbors: An integer representing the number of neighbors. Default is 6.
    
    Returns:
    A tuple containing the RMSD error, angle, and coefficients of a polynomial fit to the distortion.
    """
    RMSD_min=1e5
    angle_min=0
    angle=[]
    for j in vectors:                                           #sort vectors and angles
        angle.append(np.arctan2(j[1],j[0]))
    angle=np.array(angle)
    vectors = vectors[np.argsort(angle)]
    angle = angle[np.argsort(angle)]
    misorientation = angle - angle_ideal[0]
    maps=mappings(num_of_neighbors)[0]
    for i in range(num):
        mapping = mappings(num_of_neighbors)[i]
        theta=misorientation[i]
        rotation_matrix = get_rotation_matrix(theta)
        ideal_rotate = np.dot(ideal_vectors,rotation_matrix.T)
        RMSD=cal_RMSD(vectors,ideal_rotate,mapping,num=num_of_neighbors)
        if RMSD<RMSD_min:RMSD_min=RMSD;angle_min=theta; maps=mapping;ideal_rotate_final=ideal_rotate
    
    cand_angle_list = np.linspace(-5,5,10)*3.14/180 + angle_min;
    for theta in cand_angle_list:
        rotation_matrix = get_rotation_matrix(theta)
        ideal_rotate = np.dot(ideal_vectors,rotation_matrix.T)
        RMSD=cal_RMSD(vectors,ideal_rotate,maps,num=num_of_neighbors)
        if RMSD<RMSD_min:RMSD_min=RMSD;angle_min=theta;ideal_rotate_final=ideal_rotate
    p0=[1,0,1]
    M = optimize_M(p0,ideal_rotate_final, vectors[maps,:], num_of_neighbors=num_of_neighbors)
    R,D=decompose(M)
    a,b,c=D[0][0],D[1][1],D[1][0]
    angle_p = np.arccos(R[0,0])
    # angle_min-=angle_p
    return RMSD_min,angle_min,a,b,c


def find_median_interatomic_dist(vor):
    """
    Find the median interatomic distance from a Voronoi diagram.

    Parameters
    ----------
    vor : Voronoi object
        The Voronoi diagram.

    Returns
    -------
    median_interatomic_dist : float
        The median interatomic distance.

    """
    ridge_length = []
    for x,y in vor.ridge_points:
        length = np.linalg.norm(vor.points[x] - vor.points[y])
        ridge_length.append(length)
    ridge_length = np.array(ridge_length)
    median_interatomic_dist = np.median(ridge_length)
    return median_interatomic_dist


def get_angles(vor,ideal_vectors,angle_ideal, rotation_angle = 0, return_error = False):
    """
    Given a Voronoi diagram, ideal vectors, and an angle, returns the angles between the ideal vectors and the vectors to the 6 closest neighbors of each point in the Voronoi diagram.
    
    Args:
    vor: A Voronoi diagram object.
    ideal_vectors: A numpy array of shape (6, 2) representing the ideal vectors.
    angle_ideal: A float representing the ideal angle between the vectors.
    rotation_angle: A float representing the angle to rotate the vectors by. Default is 0.
    normalization: A boolean representing whether to normalize the vectors. Default is True.
    return_error: A boolean representing whether to return the RMSD error. Default is False.
    
    Returns:
    A numpy array of shape (npoints,) representing the angles between the ideal vectors and the vectors to the 6 closest neighbors of each point in the Voronoi diagram.
    """
    median_interatomic_dist = find_median_interatomic_dist(vor)
    #loop, sort in anti-clockwise, match, label
    angles=[]
    RMSDs=[]
    for x in tqdm(range(vor.npoints)):
#         vectors=vor.vertices[vor.regions[vor.point_region[x]]]-vor.points[x]
        vectors = get_neighbor_vectors(vor,x)
        if vectors.shape[0]!=6:
            angles.append(0);RMSDs.append(1e5);continue;
        RMSD, angle = matching(vectors,ideal_vectors,angle_ideal,3)
        angles.append(angle)
        RMSDs.append(RMSD)
    for i in range(len(angles)):
        angles[i] = angles[i]%np.pi
        angles[i] += rotation_angle
    if return_error:
        return np.array(angles), np.array(RMSDs)
    return np.array(angles)

def get_rotation_matrix(angle, if_rad = True):
    # Convert angle from degrees to radians
    if not if_rad:
        angle = math.radians(angle)
    
    # Calculate sine and cosine of the angle
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    # Construct the rotation matrix
    rotation_matrix = [
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ]
    
    return np.array(rotation_matrix)

def get_strain(points,ideal_vectors,angle_ideal, rotation_angle = 0, return_error = False, num_of_neighbors=6,angular_period=np.pi):
    """
    Given a Voronoi diagram, ideal vectors, and an ideal angle, returns the coefficients of a polynomial fit to the distortion.
    
    Args:
    vor: A Voronoi diagram.
    ideal_vectors: A numpy array of shape (6, 2) representing the ideal vectors.
    angle_ideal: A float representing the ideal angle between the vectors.
    rotation_angle: A float representing the rotation angle. Default is 0.
    return_error: A boolean representing whether to return the RMSD error. Default is False.
    num_of_neighbors: An integer representing the number of neighbors. Default is 6.
    angular_period: A float representing the angular period. Default is pi.
    
    Returns:
    A tuple containing the coefficients of a polynomial fit to the distortion.
    """ 
    #loop, sort in anti-clockwise, match, label
    vor = Voronoi(points)
    angles=np.zeros(vor.npoints)
    RMSDs=np.zeros(vor.npoints)
    a_s,b_s,c_s=np.zeros(vor.npoints),np.zeros(vor.npoints),np.zeros(vor.npoints)
    for x in tqdm(range(vor.npoints)):
        vectors = get_neighbor_vectors(vor,x,num_of_neighbors)
        if vectors.shape[0]!=num_of_neighbors:
            angles[x],a_s[x],b_s[x],c_s[x] = 0,1,1,0;RMSDs[x]=1e5;continue;
        RMSD, angle,a,b,c = matching_all(vectors,ideal_vectors,angle_ideal,3,num_of_neighbors=num_of_neighbors)
        RMSDs[x],angles[x],a_s[x],b_s[x],c_s[x] = RMSD,angle,a,b,c
    for i in range(len(angles)):
        angles[i] += rotation_angle
        angles[i] %= angular_period
    if return_error:
        return a_s,b_s,c_s,angles, RMSDs
    return a_s,b_s,c_s,angles
###################################################
""" template generation"""
###################################################
def select_region_for_FFT(img, select_area=True, plot=True):
    """
    Select a region of interest for FFT analysis.
    Parameters
    ----------
    img : numpy array
        The image to be analyzed.
    select_area : bool
        If True, the user can select a region of interest.
        If False, the whole image is used.
    Returns
    -------
    img : numpy array
        The image with the selected region of interest.
    """
    if select_area:
        roi = cv2.selectROI('roi',img, False, False)
    else:
        return img
    cv2.destroyAllWindows()
    left, right, bottom, top = roi[0], roi[0]+roi[2], roi[1], roi[1]+roi[3]
    if plot:
        img_new = deepcopy(img)
        img_new = cv2.rectangle(img_new, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (255, 0, 0), 2)
        plt.figure()
        plt.imshow(img_new)
    return img[bottom:top, left:right]

def manual_select_diff_spots(fft_img, Peak_end = False, plot=True):
    """
    Manually select regions of interest (ROIs) from an FFT image.

    Parameters
    ----------
    fft_img : numpy array
        The FFT image.
    Peak_end : bool, optional
        If True, the regions of interest are selected from the end of the FFT spectrum.
        If False, the regions of interest are selected from the start of the FFT spectrum. The default is False.

    Returns
    -------
    ROIs : list
        A list of ROIs.

    """
    if Peak_end:
        left, right, bottom, top = fft_img.shape[1]//2 - Peak_end, fft_img.shape[1]//2 + Peak_end, fft_img.shape[0]//2 - Peak_end, fft_img.shape[0]//2 + Peak_end
    else:
        left, right, bottom, top = 0, fft_img.shape[1], 0, fft_img.shape[0]
    ROIs = cv2.selectROIs('rois',fft_img[bottom:top, left:right], False, False)
    cv2.destroyAllWindows()
    img_new = deepcopy(fft_img)
    plt.figure()
    for roi in ROIs:
        roi[0] = roi[0]+bottom
        roi[1] = roi[1]+left
        img_new = cv2.rectangle(img_new, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (255, 0, 0), 2)
    if plot:
        plt.imshow(img_new[bottom:top, left:right])
    return ROIs

def get_smooth_FFT(img, sigma = 5):
    """
    Smooth an FFT image using a Gaussian filter.

    Parameters
    ----------
    img : numpy array
        The image to be filtered.
    sigma : int, optional
        The standard deviation for the Gaussian filter. The default is 5.

    Returns
    -------
    im1 : numpy array
        The filtered image.

    """
    F1 = fftpack.fft2((img).astype(float))
    F2 = fftpack.fftshift(F1)
    im1 = (20*np.log10( 0.1 + F2)).real.astype(int)
    im1 = gaussian_filter(im1,sigma) 
    im1 = rescale_intensity(im1, out_range=(0,255)).astype(np.uint8)
    return im1

def get_maximum_from_POIs(img, ROIs,plot=True):
    """
    Get the maximum points from regions of interest.

    Parameters
    ----------
    img : numpy array
        The input image.
    ROIs : list
        A list of regions of interest.

    Returns
    -------
    locs : numpy array
        The maximum points from the regions of interest.

    """
    locs = []
    fft_img = get_smooth_FFT(img, sigma = 0)
    for roi in ROIs:
        left, right, bottom, top = roi[0], roi[0]+roi[2], roi[1], roi[1]+roi[3]
        loc_fft = fft_img[bottom:top, left:right]
        max_loc = np.unravel_index(loc_fft.argmax(), loc_fft.shape)
        locs.append([max_loc[1]+left, max_loc[0]+bottom])
    locs = np.array(locs)
    if plot:
        plt.scatter(locs[:,0],locs[:,1],s=1.5, color = 'r')
    return locs

def template_map(img, Peak_end = False, select_area=True,plot = False):
    """
    Create a template map for FFT analysis.
    Parameters
    ----------
    img : numpy array
        The image to be analyzed.
    Peak_end : bool
        If True, the template map is created from the end of the FFT spectrum.
        If False, the template map is created from the start of the FFT spectrum.
    select_area : bool
        If True, the user can select a region of interest.
        If False, the whole image is used.
    Returns
    -------
    im1 : numpy array
        The template map.
    """
    img = select_region_for_FFT(img,select_area=select_area, plot=plot)
    fft_img = get_smooth_FFT(img, sigma = 0)
    ROIs = manual_select_diff_spots(fft_img, plot=plot)
    max_locs = get_maximum_from_POIs(img, ROIs, plot=plot)
    mask = np.zeros_like(fft_img)
    for i in max_locs:
        mask[i[1],i[0]] = 1
    im1 = fftpack.ifft2(fftpack.ifftshift(mask*fft_img)).real
    im1 = rescale_intensity(im1, out_range=(0, 255)).astype("uint8")
    if plot:
        print('spot of interests = ', max_locs)
    return im1,max_locs-np.array([img.shape[0]//2,img.shape[1]//2])

def get_template_factors(img_template,num_of_neighbors,neighbor_distance,given_point=None):
    """
    Given an image template and the number of neighbors, returns the ideal vectors and angles.
    
    Args:
    img_template: An image template.
    num_of_neighbors: An integer representing the number of neighbors.
    
    Returns:
    A tuple containing the ideal vectors and angles.
    """
    lattice = peak_local_max(img_template, min_distance=int(neighbor_distance/2),exclude_border=5)
    lattice = refine_atom_locs(img_template, lattice, int(neighbor_distance/1.5))
    plt.figure()
    plt.imshow(img_template)
    plt.scatter(lattice[:, 1], lattice[:, 0],s=1.5, color = 'r')
    vor = Voronoi(lattice)
    template_vectors = []
    if given_point:
        displacement = get_neighbor_vectors(vor,given_point,num_of_neighbors)
        ideal_vectors, angle_ideal = sort_angle(displacement)
        return ideal_vectors, angle_ideal
    for x in range(vor.npoints):
        displacement = get_neighbor_vectors(vor,x,num_of_neighbors)
        if (np.linalg.norm(displacement,axis=1) > neighbor_distance*1.5).sum() == 0:
            template_vectors.append(displacement)
    template_vectors = np.vstack(template_vectors)
    _, centers = cluster_and_average(template_vectors.reshape(-1,2),num_clusters=num_of_neighbors)
    ideal_vectors, angle_ideal = sort_angle(centers)
    return ideal_vectors, angle_ideal


#######################################################
""" Lattice relaxation"""
#######################################################

def get_neighbor(n,vor,num_of_neighbors=None):
    """
    Get the indices of neighboring points in a Voronoi diagram for a given point.

    Parameters
    ----------
    n : int
        The index of the point in the Voronoi diagram for which to find neighbors.
    vor : scipy.spatial.Voronoi object
        The Voronoi diagram to search for neighbors.
    num_of_neighbors : int, optional
        The maximum number of neighbors to return. If not specified, returns all neighbors.
        Default is None.

    Returns
    -------
    neighbor_candi : numpy array
        An array containing the indices of neighboring points in the Voronoi diagram.
    """
    neighbor_candi = np.concatenate([vor.ridge_points[vor.ridge_points[:,0]==n][:,1],vor.ridge_points[vor.ridge_points[:,1]==n][:,0]])
    if not num_of_neighbors:
        return neighbor_candi
    else:
        vectors = np.zeros([len(neighbor_candi),2])
        for i,index in enumerate(neighbor_candi):
            vectors[i] = vor.points[index]-vor.points[n]
        if len(neighbor_candi)>num_of_neighbors:
            sort = np.argsort(np.linalg.norm(vectors, axis=1))
            neighbor_candi = neighbor_candi[sort][:num_of_neighbors]
        return neighbor_candi
###########################################
def get_neighbor_vectors(vor,n,num_of_neighbors=6):
    """
    Given a Voronoi diagram and a point index n, returns the vectors to the n closest neighbors.
    
    Args:
    vor: A Voronoi diagram object.
    n: An integer representing the index of the point in the Voronoi diagram.
    num_of_neighbors: An integer representing the number of closest neighbors to return. Default is 6.
    
    Returns:
    A numpy array of shape (num_of_neighbors, 2) reprlattice_relaxesenting the vectors to the n closest neighbors.
    """
    idx = get_neighbor(n,vor)
    vectors = np.zeros([len(idx),2])
    for i,index in enumerate(idx):
        vectors[i] = vor.points[index]-vor.points[n]
    if len(idx)>num_of_neighbors:
        sort = np.argsort(np.linalg.norm(vectors, axis=1))
        vectors = vectors[sort][:num_of_neighbors]
    return vectors
###########################################

def get_all_edges(vor,num_of_neighbors=None):
    """
    Get all edges of the Voronoi diagram.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi object
        The Voronoi diagram for which to obtain edges.
    num_of_neighbors : int, optional
        The maximum number of neighbors to consider when finding edges.
        Default is None, which returns all edges.

    Returns
    -------
    ridges : numpy array
        An array of all edges in the Voronoi diagram.
    """
    if not num_of_neighbors:
        return vor.ridge_points
    ridges = []
    for atom in range(vor.npoints):
        neighbors = get_neighbor(atom,vor,num_of_neighbors=num_of_neighbors)
        for neighbor in neighbors:
            ridges.append([atom,neighbor])
    ridges = np.vstack(np.array(ridges,dtype=int))
    return ridges
###########################################
def get_graph_relaxation(points, bond_thres = False, node_info=None, num_of_neighbors=None):
    """
    Create a graph from a set of points using a Voronoi diagram and apply relaxation.

    Parameters
    ----------
    points : numpy array
        The points to use for the Voronoi diagram and graph.
    bond_thres : float, optional
        The threshold distance for edge removal.
        Default is False, which sets the threshold to 1.5 times the median ridge length.
    node_info : numpy array, optional
        An array containing additional information for each node in the graph.
        Default is None.
    num_of_neighbors : int, optional
        The maximum number of neighbors to consider when finding edges.
        Default is None, which returns all edges.

    Returns
    -------
    G : networkx graph
        The relaxed graph.
    """
    vor = Voronoi(points)
    G = nx.Graph()
    G.add_nodes_from(range(len(vor.points)))
    G.add_edges_from(get_all_edges(vor, num_of_neighbors))
    for i in G.nodes:
        G.nodes[i]['loc'] = points[i]
        G.nodes[i]['original_loc'] = points[i]
        
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
###########################################
def image_force(image, loc):
    """
    Calculate the net force exerted by an image on a location.

    Parameters
    ----------
    image : numpy array
        The image on which to calculate the force.
    loc : tuple of int
        The location at which to calculate the force.

    Returns
    -------
    net_force : numpy array
        The net force exerted on the location by the image.
    """
    net_force = np.zeros(2)
    sobel_kernel_x = -1*np.array([[1,0,-1],[2,0,-2],[1,0,-1]]).T
    sobel_kernel_y = -1*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    local_area = image[loc[0]-1:loc[0]+2,loc[1]-1:loc[1]+2]
    net_force[0] = (sobel_kernel_x*local_area).sum()
    net_force[1] = (sobel_kernel_y*local_area).sum()
    return net_force

def image_force_2(image, loc, original_loc):
    """
    Calculate the net force exerted by the original location.

    Parameters
    ----------
    image : numpy array
        The image on which to calculate the force.
    loc : tuple of int
        The location at which to calculate the force.
    original_loc : tuple of int
        The original location at which to calculate the force.
    Returns
    -------
    net_force : numpy array
        The net force exerted on the location by the image.
    """
    net_force = np.array(original_loc) - np.array(loc)
    return net_force
###########################################
def bond_force(G, node, thres):
    """
    Calculate the bond force exerted on a node by its edges.

    Parameters
    ----------
    G : networkx graph
        The graph containing the node and its edges.
    node : int
        The node for which to calculate the bond force.
    thres : float
        The threshold bond force value.

    Returns
    -------
    net_force : numpy array
        The net force exerted on the node by its edges.
    """
    net_vector = np.zeros(2)
    for n1, n2, data in G.edges(node, data=True):
        # Calculate the force exerted by the spring using Hooke's law
        spring_vector = G.nodes[n2]['loc'] - G.nodes[n1]['loc']
        # Add the spring force to the net force
        net_vector += spring_vector
    if np.linalg.norm(net_vector)>thres:
        return 0
    net_force = data['k'] * (net_vector - data['l0']*net_vector/np.linalg.norm(net_vector))
    return net_force
###########################################
def lattice_relax(G, image, epoch, beta, step, max_bond_force, median_interatomic_dist, neighbor_num=6):
    """
    Perform lattice relaxation on a graph.

    Parameters
    ----------
    G : networkx graph
        The graph to relax.
    image : numpy array
        The image on which to calculate forces.
    epoch : int
        The number of iterations to perform for relaxation.
    beta : float
        The weighting factor for the bond force.
    step : float
        The step size for each iteration.
    max_bond_force : float
        The maximum bond force value.
    neighbor_num : int, optional
        The number of neighbors to use for relaxation.
        Default is 6.

    Returns
    -------
    disp : list of numpy array
        The displacement of each iteration.
    """
    disp = []
    for i in G.nodes:
        G.nodes[i]['bond_force'] = bond_force(G,i,thres=median_interatomic_dist/2)
    for i in G.nodes:
        loc = G.nodes[i]['loc']
        G.nodes[i]['image_force'] = image_force_2(image,loc)/10
    for _ in tqdm(range(epoch)):
        total_disp = np.zeros(2)
        for i in G.nodes:
            loc = G.nodes[i]['loc']
            original_loc = G.nodes[i]['original_loc']
            G.nodes[i]['bond_force'] = bond_force(G,i,thres=median_interatomic_dist/2) if np.linalg.norm(bond_force(G,i,thres=median_interatomic_dist/2)) < max_bond_force else max_bond_force*bond_force(G,i,thres=median_interatomic_dist/2)/np.linalg.norm(bond_force(G,i,thres=median_interatomic_dist/2))
            G.nodes[i]['image_force'] = image_force_2(image,loc,original_loc)
            G.nodes[i]['net_force'] = beta*G.nodes[i]['bond_force'] + G.nodes[i]['image_force']/20
        for i in G.nodes:
            if len(list(G.neighbors(i))) != neighbor_num:
                continue;
            G.nodes[i]['loc'] = G.nodes[i]['loc'] + G.nodes[i]['net_force']*step
#             if np.isnan(G.nodes[i]['loc'][0]):
#                 print(G.nodes[i]['bond_force'], G.nodes[i]['image_force'])
            total_disp = total_disp + np.linalg.norm(G.nodes[i]['net_force']*step)
        disp.append(total_disp)
    return disp
###########################################


def plot_strain(image,coordinates_all_plot,angles,a,b,c, save_path = False, plot = True, mask=False, plot_range = 0.1):
    dots = coordinates_all_plot
    if isinstance(mask, bool):
        mask = np.ones(dots.shape[0],dtype=bool)
    fig, ax = plt.subplots(figsize=(10,8),ncols=2,nrows=2,dpi = 200, sharex=True,sharey=True)
    ax[0][0].imshow(image)
    sc=ax[0][0].scatter(dots[mask,1],dots[mask,0],s=2,c=angles[mask],cmap='hsv')
    vmin,vmax = np.pi/6-0.05,np.pi/6+0.05
    # sc.set_clim(vmin,vmax)
    divider = make_axes_locatable(ax[0][0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax, orientation='vertical')

    ax[0][1].imshow(image)
    sc=ax[0][1].scatter(dots[mask,1],dots[mask,0],s=5,c=a[mask]-1,cmap='inferno')
    sc.set_clim(-plot_range,plot_range)
    divider = make_axes_locatable(ax[0][1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax, orientation='vertical')


    ax[1][0].imshow(image)
    sc=ax[1][0].scatter(dots[mask,1],dots[mask,0],s=5,c=b[mask]-1,cmap='inferno')
    sc.set_clim(-plot_range,plot_range)
    divider = make_axes_locatable(ax[1][0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax, orientation='vertical')


    ax[1][1].imshow(image)
    sc=ax[1][1].scatter(dots[mask,1],dots[mask,0],s=5,c=c[mask],cmap='inferno')
    sc.set_clim(-plot_range,plot_range)
    divider = make_axes_locatable(ax[1][1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax, orientation='vertical')
    plt.tight_layout()
    if isinstance(save_path, bool):
        return None
    else:
        plt.savefig(save_path)
    if plot == False:
        plt.clf()