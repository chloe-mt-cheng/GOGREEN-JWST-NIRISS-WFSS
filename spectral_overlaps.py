#Imports
import numpy as np
import os
from astropy.io import fits
from astropy import units as u
import itertools
import astropy.coordinates as coord
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

font = {'family':'serif',
        'weight':'normal',
        'size':20
}
plt.rc('font',**font)


def all_coordinates(rectangles):
    """Return the coordinates of the 4 vertices of a set of rectangles.

    Parameters
    ----------
    rectangles : lst
        A list of matlplotlib.patches.Rectangle objects

    Returns
    -------
    all_coords : tuple
        An array of arrays of 4 sets of coordinates for the vertices of each rectangle
    """

    all_coords = []
    for i in range(len(rectangles)):
        all_coords.append(rectangles[i].get_patch_transform().transform(rectangles[i].get_path().vertices[:-1]))
    all_coords = np.array(all_coords)
    return all_coords

def rectangle_vertices(coordinates):
    """Return the (x,y) of the vertices of a rectangle

    Parameters
    ----------
    coordinates : tuple
        A set of 4 coordinates defining the vertices of a rectangle

    Returns
    -------
    bottom_left : tuple
        The coordinates of the bottom-left corner of the rectangle
    bottom_right : tuple
        The coordinates of the bottom-right corner of the rectangle
    top_right : tuple
        The coordinates of the top-right corner of the rectangle
    top_left : tuple
        The coordinates of the top-left corner of the rectangle
    """

    bottom_left = coordinates[0]
    bottom_right = coordinates[1]
    top_right = coordinates[2]
    top_left = coordinates[3]
    return bottom_left, bottom_right, top_right, top_left

def intersection_area(coordinates1, coordinates2):
    """Return the area of intersection between two rectangles.

    Parameters
    ----------
    coordinates1 : tuple
        A set of 4 coordinates defining the vertices of the first rectangle
    coordinates2 : tuple
        A set of 4 coordinates defining the vertices of the second rectangle

    Returns
    -------
    area : float
        The area of intersection between the two rectangles
    """

    bottom_left1, bottom_right1, top_right1, top_left1 = rectangle_vertices(coordinates1)
    bottom_left2, bottom_right2, top_right2, top_left2 = rectangle_vertices(coordinates2)

    top_right_x1 = top_right1[0]
    top_right_y1 = top_right1[1]
    bottom_left_x1 = bottom_left1[0]
    bottom_left_y1 = bottom_left1[1]

    top_right_x2 = top_right2[0]
    top_right_y2 = top_right2[1]
    bottom_left_x2 = bottom_left2[0]
    bottom_left_y2 = bottom_left2[1]

    intersect_top_right_x = np.min((top_right_x1, top_right_x2))
    intersect_top_right_y = np.min((top_right_y1, top_right_y2))
    intersect_bottom_left_x = np.max((bottom_left_x1, bottom_left_x2))
    intersect_bottom_left_y = np.max((bottom_left_y1, bottom_left_y2))

    area = np.max((0, (intersect_top_right_y - intersect_bottom_left_y)))*np.max((0, (intersect_top_right_x -
                                                                                      intersect_bottom_left_x)))
    return area

def percentage_overlap(lengths, heights, mags, rectangles, coordinates, overlap_target, brightness_target):
    """Return the percentage of rectangles in the FOV that are overlapping by greater than the
    target overlap percentage, where the two objects are brightness_target times as bright as each other.

    Parameters
    ----------
    lenghts : tuple
        A set of spectral lengths
    heights : tuple
        A set of object sizes
    mags : tuple
        The K-magnitudes of the objects in the cluster
    rectangles : list
        A list of matplotlib.patches.Rectangle objects
    coordinates : tuple
        A set of 4 coordinates for each rectangle
    overlap_target : float
        The fraction by which rectangles overlap
    brightness_target : float
        The fraction by which overlapping rectangles are brightness_target times as bright as each other.

    Returns
    -------
    total_percentage_overlap : float
        The percentage of rectangles that are overlapping in the FOV with the given overlap_target and
        brightness_target
    """

    #Area of each rectangle
    rectangle_areas = lengths*heights

    #Compute the intersection areas, union areas, and magnitude ratios
    intersection_areas = []
    compared_mags = []
    inds = []
    union_areas = []
    for i in itertools.permutations(np.arange(0, len(rectangles)), 2):
        intersection_areas.append(intersection_area(coordinates[i[0]], coordinates[i[1]]))
        compared_mags.append(mags[i[0]]/mags[i[1]])
        inds.append(i)
        union_areas.append(rectangle_areas[i[0]] + rectangle_areas[i[1]] -
                           intersection_area(coordinates[i[0]], coordinates[i[1]]))
    intersection_areas = np.array(intersection_areas)
    union_areas = np.array(union_areas)

    #Compute the percentage overlap for each set of rectangles
    percentage_overlap = intersection_areas/union_areas

    #Find out which combinations of rectangles have overlaps of > 20% with the magnitude of one rectangle 0.1x as
    #bright as the other
    targ_inds = []
    for i in range(len(percentage_overlap)):
        if percentage_overlap[i] > overlap_target and compared_mags[i] >= brightness_target:
            targ_inds.append(inds[i])

    #Remove duplicates so we just get the number of overlapping rectangles with these characteristics
    true_inds = np.array(targ_inds).flatten()
    final_rects = []
    [final_rects.append(x) for x in true_inds if x not in final_rects]
    final_rects = np.array(final_rects)

    #Get the total percentage of overlapping rectangles
    total_percentage_overlap = (len(final_rects)/len(rectangles))*100

    return total_percentage_overlap

def cluster_overlaps(name, centre_ra, centre_dec, ext, overlap_target, brightness_target):
    """Return the overlap percentage for a particular cluster.  Also make a plot of the spectral overlaps.

    Parameters
    ----------
    name : str
        The name of the cluster as defined in the photo.fits catalogue (i.e. 'SPT0205')
    centre_ra : float
        The right ascension of the centre of the cluster
    centre_dec : float
        The declination of the centre of the cluster
    ext : str
        The extension by which to save the plot
    overlap_target : float
        The fraction by which rectangles should be overlapping
    brightness_target : float
        The fraction by which overlapping rectangles should be brightness_target times as bright as each other

    Returns
    -------
    overlaps : float
        The percentage by which the sources in the FOV are overlapping with the given overlap_target and
        brightness_target 
    """

    #Read in GOGREEN data
    file = os.path.abspath('Photo.fits')
    hdul = fits.open(file)
    header_info = hdul[1].header
    ids = hdul[1].data['cPHOTID']
    clusters = hdul[1].data['Cluster']
    ra_list = hdul[1].data['ra']
    dec_list = hdul[1].data['dec']
    Kmag = hdul[1].data['Ks_tot']
    hdul.close()

    #Get data for cluster
    cluster_inds = np.argwhere(clusters == name).squeeze()
    cluster_ra = ra_list[cluster_inds]
    cluster_dec = dec_list[cluster_inds]
    cluster_Kmag = Kmag[cluster_inds]
    cluster_ids = ids[cluster_inds]

    #Get cluster number
    name_digits = ''
    for i in name:
        if i.isdigit():
            name_digits += i
    #Get name in file
    if 'SPT' in name:
        proper_name = 'SPTCL-' + name_digits
    else:
        proper_name = 'SpARCS-' + name_digits

    #Read in the photometry catalogue for the cluster
    file2 = os.path.abspath(proper_name + '_totalall_FOURSTARKs.cat')
    hdul2 = pd.read_table(file2, sep='\t|\s+')

    #Calculate x- and y-pixel offsets from the centre of the cluster
    plate_scale = 0.065
    cluster_coords = coord.SkyCoord(ra=cluster_ra*u.deg, dec=cluster_dec*u.deg, frame='icrs')
    centre_coords = coord.SkyCoord(ra=centre_ra*u.deg, dec=centre_dec*u.deg, frame='icrs')
    ra_offsets = ((cluster_coords.ra - centre_coords.ra)*np.cos(centre_coords.dec.to(u.radian))).to(u.arcsec)
    dec_offsets = (cluster_coords.dec - centre_coords.dec).to(u.arcsec)
    x_offsets = (ra_offsets/plate_scale).value
    y_offsets = (dec_offsets/plate_scale).value

    #Find the sources in the NIRISS FOV
    niriss_fov = (2.2*u.arcmin.to(u.arcsec))/plate_scale
    inds_inrange = (np.abs(x_offsets) < niriss_fov) & (np.abs(y_offsets) < niriss_fov)
    xpix_inrange = x_offsets[inds_inrange]
    ypix_inrange = y_offsets[inds_inrange]
    kmags_inrange = cluster_Kmag[inds_inrange]
    ids_inrange = cluster_ids[inds_inrange]

    #Turn all IDs into strings
    ids_inrange_str = []
    for i in range(len(ids_inrange)):
        ids_inrange_str.append(str(ids_inrange[i]))

    #Get the half-light radii
    half_light_rad = []
    ids_test = []
    for i in range(len(ids_inrange_str)):
        if ids_inrange_str[i][-4] == str(0):
            for j in range(len(hdul2['id'])):
                if str(hdul2['id'][j]) == ids_inrange_str[i][-3:]:
                    half_light_rad.append(hdul2['K_fluxrad'][j])
                    ids_test.append(hdul2['id'][j])
        elif ids_inrange_str[i][-5] == str(0):
            for j in range(len(hdul2['id'])):
                if str(hdul2['id'][j]) == ids_inrange_str[i][-4:]:
                    half_light_rad.append(hdul2['K_fluxrad'][j])
                    ids_test.append(hdul2['id'][j])
    half_light_rad = np.array(half_light_rad)
    all_object_size = half_light_rad*2 #Multiply by 2 to get the diameter (?)

    #Get the spectral lengths
    dispersion = 0.00478 #NIRISS dispersion, um/pixel
    wave_bottom = 1.3 #um
    wave_top = 1.7 #um
    wave_range = wave_top - wave_bottom
    spec_length = np.full(len(xpix_inrange), wave_range/dispersion) #in pixels

    #Plot the overlaps for visualization
    fig, ax = plt.subplots(1, figsize=(10,8))
    fig.suptitle('Spectral Overlap of GR150C F150W in ' + name, fontsize=20)
    ax.plot(xpix_inrange, ypix_inrange, linestyle='None')
    rectangles = []
    for i in range(len(xpix_inrange)):
        rectangles.append(patches.Rectangle(xy=(xpix_inrange[i], ypix_inrange[i]), width=spec_length[i],
                                                height=all_object_size[i], fill=False, linewidth=3, alpha=0.7, edgecolor='b'))
        ax.add_patch(rectangles[i])
    ax.set_xlabel('Spectral Length (Pixels)', fontsize=15)
    ax.set_ylabel('Object Size (Pixels)', fontsize=15)
    ax.tick_params(axis='both', length=6, width=2, labelsize=12)
    plt.savefig('spec_overlap_' + name + ext, bbox_inches='tight')

    #Calculate percentage overlap
    all_coords = all_coordinates(rectangles)
    overlaps = percentage_overlap(spec_length, all_object_size, kmags_inrange, rectangles, all_coords, overlap_target,
                                 brightness_target)
    return overlaps
