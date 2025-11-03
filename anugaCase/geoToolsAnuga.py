''' 
Different tools for dealing with data for creation of Anuga domain
'''


import geopandas as gpd

import math
import matplotlib.pyplot as plt

from typing import List, Tuple

from shapely.geometry import LineString, Point, Polygon

from typing import List

import numpy as np


def get_bc_masks(BcsPol, updatedBoundary, buffer_distance=0.0):
    """
    For each polygon in BcsPol, find the indices of the first and last points in updatedBoundary
    that are within the given buffer_distance of the polygon (distance <= buffer_distance).
    This includes points inside, on the boundary, or outside but within the distance.
    Assumes the relevant points are consecutive and returns a list of [start_index, end_index] pairs.
    
    :param BcsPol: List of polygons, each as a list of (x, y) coordinates.
    :param updatedBoundary: List of (x, y) points forming the boundary polygon.
    :param buffer_distance: The maximum distance to consider a point as 'within' the polygon. Default 0.0.
    :return: List of [start, end] index pairs, one for each polygon in BcsPol.
    """
    masks = []
    for bc_coords in BcsPol:
        poly = Polygon(bc_coords)
        indices = [
            i for i, coord in enumerate(updatedBoundary) 
            if Point(coord).distance(poly) <= buffer_distance
        ]
        if indices:
            masks.append([min(indices), max(indices)])
        else:
            masks.append([])  # Empty list if no points are found
    return masks


def extract_coordinates(geometry):
    if geometry.geom_type == 'Polygon':
        return list(geometry.exterior.coords)
    elif geometry.geom_type == 'MultiPolygon':
        return [list(poly.exterior.coords) for poly in geometry.geoms]
    elif geometry.geom_type == 'LineString':
        return list(geometry.coords)
    elif geometry.geom_type == 'MultiLineString':
        return [list(line.coords) for line in geometry.geoms]
    elif geometry.geom_type == 'Point':
        return [[geometry.x, geometry.y]]
    else:
        return []




def featuresToPolylines(featureFile:str,field: str | None = None) ->  List[List[List[float]]]:
    ''' 
    featuresToPolylines(featureFile:str) -> list
    featuresToPolylines(featureFile:str,field:str = None) -> Tuple[list,list]

    extracts coordinates of a geo features to a format:
    [ 
      [[x1,y1],[x2,y2],...] # f
      [[x1,y1],[x2,y2],...]
    ]
    '''

    tab = gpd.read_file(featureFile)

    coordinates_list = []
       

    for geom in tab.geometry:
        coords = extract_coordinates(geom)
        # Flatten if it's a list of lists (e.g., MultiPolygon)
        if isinstance(coords[0][0], (float, int)):
            coordinates_list.append(coords)
        else:
            coordinates_list.extend(coords)
    
    # for args loop:

    if field is not None:
        args_list = list(tab[field])
        return coordinates_list, args_list
    else:
        return coordinates_list



# copilot solution


def find_nearby_indices(boundary: List[Tuple[float, float]], bcLine: List[List[float]], radius: float) -> List[int]:
    bc_line_geom = LineString(bcLine)
    nearby_indices = []
    for i, point in enumerate(boundary):
        pt = Point(point)
        if pt.distance(bc_line_geom) <= radius:
            nearby_indices.append(i)
    return nearby_indices






## for debuging only

def plot_polylines_with_labels(boundary, bcLine):
    """
    Plot boundary points with index labels and bcLine polyline.
    
    Args:
        boundary: List of (x, y) tuples representing the boundary polyline
        bcLine: List of [x, y] lists representing the reference polyline
    """
    # Extract x, y coordinates from boundary
    boundary_x = [point[0] for point in boundary]
    boundary_y = [point[1] for point in boundary]
    
    # Extract x, y coordinates from bcLine
    bcLine_x = [point[0] for point in bcLine]
    bcLine_y = [point[1] for point in bcLine]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot boundary points as scatter with labels
    plt.scatter(boundary_x, boundary_y, color='blue', label='Boundary Points', s=50)
    for i, (x, y) in enumerate(boundary):
        plt.text(x, y, str(i), fontsize=8, ha='right', va='bottom')
    
    # Plot bcLine as a connected line
    plt.plot(bcLine_x, bcLine_y, color='red', label='bcLine', linewidth=2)
    
    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Boundary Points and bcLine')
    plt.legend()
    plt.grid(True)
    
    # Adjust plot limits to show all points clearly
    # x_range = max(max(boundary_x), max(bcLine_x)) - min(min(boundary_x), min(bcLine_x))
    # y_range = max(max(boundary_y), max(bcLine_y)) - min(min(boundary_y), min(bcLine_y))
    # plt.margins(x=x_range*0.05, y=y_range*0.05)
    
    plt.show()


def find_vector_intersection(vector1, vector2):
    """
    Find intersection point(s) between two polylines with multiple points.
    
    Args:
        vector1: List of points [[x1,y1], [x2,y2], ..., [xN,yN]] defining first polyline
        vector2: List of points [[x1,y1], [x2,y2], ..., [xM,yM]] defining second polyline
    
    Returns:
        List of intersection points [[x,y], ...] or empty list if no intersection
    """
    # Validate input
    if not (len(vector1) >= 2 and len(vector2) >= 2):
        raise ValueError("Each vector must have at least two points")
    if not all(len(point) == 2 for point in vector1 + vector2):
        raise ValueError("Each point must have exactly two coordinates [x,y]")

    # Create LineString objects from input coordinates
    line1 = LineString(vector1)
    line2 = LineString(vector2)
    
    # Find intersection
    intersection = line1.intersection(line2)
    
    # Handle different intersection types
    result = []
    
    if intersection.is_empty:
        return result
    elif intersection.geom_type == 'Point':
        result.append([intersection.x, intersection.y])
    elif intersection.geom_type == 'MultiPoint':
        for point in intersection.geoms:
            result.append([point.x, point.y])
    
    return result

# put points on the vector




def includePointToVector(baseVector:list, point:list) -> Tuple[list,list]:
    '''
    puts a point (x,y) ist on the baseVector. 

    returns modified baseVector and indexes of new points
    beaseVector, points format: [[x1,y1], [x2,y2], ..., [xN,yN]]

    Important, assume that points are on the baseVactor polyline! 
    e.g. by find_vector_intersection

    '''
    xp,yp = point # extracting coords of a point

    newBaseVector = []
    newPointsIndexes = []

    ind = 0
    for k in range(len(baseVector)):
        xb0, yb0 = baseVector[k-1] # starts from -1!
        xb1, yb1 = baseVector[k]

        # calculate distance from xb0, yb0  to xp,yp
        d0 = math.sqrt((xb0-xp)**2 +  (yb0-yp)**2  )
        
        # calculate distance from xb1, yb1  to xp,yp
        d1 = math.sqrt((xb1-xp)**2 +  (yb1-yp)**2  )

        # calculate distance from xb1, yb1  to xb0, yb0
        d = math.sqrt((xb1-xb0)**2 +  (yb1-yb0)**2  )

        if d0 <= d and d1<=d:
            newBaseVector.append([xp,yp])
            newPointsIndexes.append(ind)
            ind += 1

        # org
        newBaseVector.append([xb1,yb1])
        ind += 1


    return newBaseVector,newPointsIndexes



def includePointsToVector(baseVector:list, points:list) -> Tuple[list,list]:
    '''
    puts a points from points list on the baseVector. 

    returns modified baseVector and indexes of new points 
    
    beaseVector, points format: [[x1,y1], [x2,y2], ..., [xN,yN]]

    Important, assume that points are on the baseVactor polyline! 
    e.g. by find_vector_intersection

    '''

    newBaseVector = baseVector
    indicies = []
    for pt in points:
        newBaseVector,inds = includePointToVector(newBaseVector,pt)
        indicies.extend(inds)

    return newBaseVector, indicies


def interpolate_polyline(points:list, d:float) -> list:
    '''
    interpolates points along lines
    '''

    if not points or d <= 0:
        return []
    
    result = [points[0]]
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len = math.sqrt(dx**2 + dy**2)
        
        if seg_len > 0:
            num_added = int(seg_len // d)
            for k in range(1, num_added + 1):
                pos = k * d
                if pos < seg_len:
                    frac = pos / seg_len
                    new_x = p1[0] + frac * dx
                    new_y = p1[1] + frac * dy
                    result.append((new_x, new_y))
            result.append(p2)
    
    return result


def signedArea(P:list)->float:
    """
    Calculates signed area using Shoelace Formula

    if clockwise signed_area<0, if counterclockwise: signed_area>0,

    by grok: https://grok.com/share/bGVnYWN5_1fc806b6-863b-4246-a5c5-6320505b7372
    """
    n = len(P)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += P[i][0] * P[j][1] - P[j][0] * P[i][1]
    
    return signed_area/2




def simplify_polygon_coords(coords_list: List[List[float]], tolerance: float = 1.0) -> List[List[float]]:
    """
    Simplify a list of polygon coordinates (e.g., from featuresToPolylines) using Douglas-Peucker.
    
    :param coords_list: List of [[x1,y1], [x2,y2], ...] (single polygon) or list of such lists (multi).
    :param tolerance: Distance threshold for simplification (m). Higher = coarser.
    :return: Simplified coordinates.
    """
    simplified = []
    for coords in coords_list if isinstance(coords_list[0], list) and len(coords_list) > 1 else [coords_list]:
        if len(coords) < 3:
            simplified.append(coords)  # Can't simplify tiny polygons
            continue
        poly = Polygon(coords)
        simplified_poly = poly.simplify(tolerance, preserve_topology=True)
        if simplified_poly.is_empty:
            simplified.append(coords)  # Fallback if simplification fails
        else:
            simplified_coords = list(simplified_poly.exterior.coords)[:-1]  # Remove duplicate last point
            simplified.append(simplified_coords)
    return simplified[0] if len(simplified) == 1 else simplified

def polygonToRaster(poly: List[List[float]], x_min:float,y_min:float, x_max:float,y_max:float,dx:float,dy:float,dType:np.float32) -> np.ndarray:
    """
    Convert a polygon to a rasterized binary mask on a regular grid.

    This function takes a polygon defined by a list of coordinate pairs and
    rasterizes it into a 2D NumPy array where grid cells inside the polygon
    are set to 1 (or equivalent in the specified dtype) and outside are 0.
    It uses Matplotlib's Path for containment checks, making it suitable for
    spatial masking in GIS, image processing, or simulations.

    Parameters
    ----------
    poly : List[List[float]]
        A list of lists representing the polygon's vertices as [[x1, y1], [x2, y2], ..., [xn, yn]].
        The polygon should ideally be closed (first and last points the same), but Path handles open polygons.
        Assumes counter-clockwise ordering for proper interior detection.
    x_min : float
        The minimum x-coordinate for the grid extent.
    y_min : float
        The minimum y-coordinate for the grid extent.
    x_max : float
        The maximum x-coordinate for the grid extent.
    y_max : float
        The maximum y-coordinate for the grid extent.
    dx and dy : float
        The resolution (step size) of the grid in both x and y directions. Smaller values
        yield higher-resolution rasters but increase computation time and memory usage.
    dType : type, optional
        The NumPy data type for the output array (default: np.float32). Common choices include
        np.int8 or np.bool_ for binary masks, or np.float32 for floating-point values.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where 1 indicates points inside the polygon and 0 outside.
        Shape is approximately ((y_max - y_min) / grid_size, (x_max - x_min) / grid_size).

    Examples
    --------
    >>> square_poly = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
    >>> raster = polygonToRaster(square_poly, -5.0, -5.0, 15.0, 15.0, 1.0, 1.0, np.int8)
    >>> print(raster.shape)
    (21, 21)

    Notes
    -----
    - For performance with large grids, consider alternatives like rasterio or shapely if needed.
    - Points on the boundary may vary in inclusion; use contains_points(radius=0) for strict checks if required.
    - Ensure poly has at least 3 points; add poly.append(poly[0]) if not closed.

    See Also
    --------
    matplotlib.path.Path : For advanced path operations.
    numpy.meshgrid : For understanding grid creation.
    """

    # dx = grid_size
    xnew = np.arange(x_min, x_max + dx, dx)
    ynew = np.arange(y_min, y_max + dy, dy)
    xx, yy = np.meshgrid(xnew, ynew)

    
    path = Path(poly)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    mask = path.contains_points(points)
    return mask.reshape(xx.shape).astype(dType)