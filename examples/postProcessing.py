import anuga
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


from geoToolsAnuga import featuresToPolylines


domainName = "KozieniceTest_r400"

boundaryFile = "shp/anugaModelBoundary.gpkg"



boundary = featuresToPolylines(boundaryFile)

boundaryNp = np.array(boundary[0])


domain = anuga.file.sww.Xload_sww_as_domain(domainName+".sww")



anuga.plot_utils.Make_Geotif(domainName+".sww", 
                    output_quantities=['depth', 'velocity'],
                    myTimeStep='last',
                    CellSize=10.0,
                    EPSG_CODE = 2180,
                    bounding_polygon=boundary[0],
                    k_nearest_neighbours=1)

# anuga.plot_utils.Make_Geotif(domainName+".sww", 
#                     output_quantities=['xmomentum'],
#                     myTimeStep='last',
#                     CellSize=10.0,
#                     EPSG_CODE = 2180,
#                     bounding_polygon=boundary[0],
#                     k_nearest_neighbours=1)

# domain.get_quantity("xmomentum").get_maximum_value()




# jeszcze do dorato:

# https://github.com/ornldaac/deltax_workshop_2024/blob/main/tutorials/2_SedimentTransport_Dorado/ex2_dorado_unsteady.ipynb

data = Dataset(domainName+".sww")
print('This is the data structure of the ANUGA output file:', data)

x_min = data.xllcorner
y_min = data.yllcorner


# Extract the time and coordinates of the grid saved as distance from the lower left corner.





# używam anugi by wyciągnąć raster
swwvals = anuga.plot_utils.get_centroids(domainName+".sww", timeSlices = 'all')

time = swwvals.time
x = swwvals.x + x_min
y = swwvals.y + y_min

# Finally we make a list with the cell points called 'coordinates'
coordinates = [(x[i], y[i]) for i in list(range(len(x)))]

# water depth
depth = swwvals.height

# water level
stage = swwvals.stage

# discharge x-direction
qx = swwvals.xmom

# discharge y-direction
qy = swwvals.ymom

# extract the bed elevations
elev = swwvals.elev

xvel = swwvals.xvel
yvel = swwvals.yvel
# zapis wyników do npz:


np.savez_compressed(domainName + ".npz",time=time,x=x,y=y,coordinates=coordinates,depth=depth,stage=stage,qx=qx,qy=qy,elev=elev, xvel=xvel, yvel=yvel)