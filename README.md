# anugaCase

A Python library for parsing, processing, and creating domains for ANUGA 2D hydrodynamic models. It includes geometry tools for handling shapefiles, polylines, and boundaries, as well as utilities for setting up simulation cases with initial conditions, inflows, and boundary tags.

## Features
- Extract and manipulate geospatial features (polygons, lines) from files like GeoPackages or Shapefiles.
- Simplify polygons for efficient meshing.
- Create ANUGA domains from DTM rasters, boundary polygons, and interior holes.
- Assign boundary conditions, inflows, and quantities like elevation, friction, and stage.
- Tools for intersection detection, point insertion on vectors, and signed area calculation (for orientation).

## Installation
Clone the repo and install as a package:

```bash
git clone https://github.com/akiczko/anugaCase.git
cd anugaCase
pip install .
```

### Dependencies
- ANUGA (>=3.2.0; tested with 3.2.0) – Official GitHub: https://github.com/GeoscienceAustralia/anuga_core (works with Python 3.12)
- GeoPandas
- Matplotlib
- NumPy
- Shapely
- GDAL


## Usage
### Geometry Tools (`geoToolsAnuga.py`)
Utilities for working with geospatial data.

Example: Extract polylines from a GeoPackage:
```python
from anugaTools.geoToolsAnuga import featuresToPolylines

boundary_coords = featuresToPolylines("path/to/boundary.gpkg")
print(boundary_coords)  # List of [[x1,y1], [x2,y2], ...]
```

Other functions:
- `get_bc_masks`: Find boundary point indices near polygons.
- `find_vector_intersection`: Detect intersections between polylines.
- `includePointsToVector`: Insert points onto a base vector (assumes points lie on it).
- `simplify_polygon_coords`: Simplify polygons using Douglas-Peucker algorithm.
- `signedArea`: Compute signed area to determine polygon orientation (positive for counterclockwise).

### Case Creation (`createCaseAnuga.py`)
Class for setting up an ANUGA simulation domain.

Example:
```python
from anugaCase.createCaseAnuga import createCase

# Initialize with file paths and parameters
case = createCase(
    dtmName="dtm/bathymetry.tif",
    domainExtentGeoFilename="shp/domain_boundary.gpkg",
    inletShpName="shp/inlet_line.gpkg",
    roughnessRstOrValue=0.03,  # Or path to raster
    initialStage=200.0,
    meshMaxSize=100.0,
    domainName="MyDomain",
    boundaryConditionsZonesGeoName="shp/bc_zones.gpkg",
    boundaryConditionsTagsField="Tag",
    interiorHolesName="shp/holes.gpkg"  # Optional
)

# Create the domain (reads files, sets elevation/friction/stage)
case.createDomain()

# Set boundary conditions
case.setBoundaryConditions(inflowTag="inlet", outflowTag="outlet", outflowVal=150.0)

# Set inflow
case.setInflowLine(Q=500.0)  # Discharge in m³/s

# Now run your ANUGA simulation with case.domain
```

Notes:
- Supports raster DTMs (auto-converts non-ASC to ASC temporarily).
- Boundary tags are auto-assigned based on zones; defaults to reflective.
- Polygons can be simplified for performance (tolerance based on mesh size).

## Contributing
Pull requests welcome! For issues, open a GitHub issue.

## License
MIT License (see LICENSE file).




