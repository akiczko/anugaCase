from turtle import settiltangle
from Cython import nonecheck
from traitlets import Float
import anuga
import geopandas as gpd

import math
import matplotlib.pyplot as plt

from typing import List, Tuple

import numpy as np

import os

from geoToolsAnuga import featuresToPolylines, includePointsToVector, find_vector_intersection, get_bc_masks, signedArea, simplify_polygon_coords

from singleBandGdalTools import convert_dtm_to_asc



class createCase:
    ''' 
    Klasa zawiera funkcję 
    1) tworzącą domenę na podstawie siatki prostokątnej
    3) określająca warunki początkowe
    4) przypisujące dopływ
    5) przypisującą względem tagów warunki brzegowe

    Dla obliczeń równoległych, konstruktor tylko ustawia zmienne 
    '''
    
    def __init__(self,dtmName:str,
                 domainExtentGeoFilename:str,
                 inletShpName:str,
                 roughnessRstOrValue : str | float,
                 initialStage :float,
                 meshMaxSize: float= 100,
                 domainName:str = "domain",
                 boundaryConditionsZonesGeoName:str | None = None, 
                 boundaryConditionsTagsField:str = "Tag",
                 interiorHolesName = None,
                 simplifyPolygons: bool = True):
        ''' 
        # Caution: 
        - boundaryConditionsZonesGeo should not intersect domainExtentGeo to close 
        as it force mesh to be finer. It should be larger tha goal mesh size
        - inletShpName a linestring feature - first feature is used
        - boundaryConditionsZonesGeoName - polygon, Tags stored in field boundaryConditionsTagsField 
          will be attributed to the domain extend polygon points that falls within a tagged zones
        - simplifyPolygons - simplifies polygons be setting tol=maxMeshSize**0.5 - can be overloaded by:
           self.setSimplifyTolerance()
         
        
        # Example
        ```python
        boundaryFile = "shp/anugaModelBoundary.gpkg"
        bcFile = "shp/anugaModelBcPolygons.gpkg"
        dtmName= "dtm/bathymetry2.tif"
        
        case = createCase(dtmName, boundaryFile, inletFile,
                            roughnessRstOrValue = 0.03,
                            initialStage = 200,
                          meshMaxSize=100,
                          domainName="Domena",
                          boundaryConditionsZonesGeoName = bcFile,
                          boundaryConditionsTagsField = "Tag")


        ```
        '''

        self.settingsDict = {"DTM":dtmName,
                              "Boundaries": domainExtentGeoFilename,
                              "InletLine": inletShpName,
                              "boundaryConditionsZonesGeoName" :boundaryConditionsZonesGeoName,
                              "boundaryConditionsTagsField" :boundaryConditionsTagsField,
                              "domainName":domainName,
                              "roughness" : roughnessRstOrValue,
                              "initialStage": initialStage,
                              "interiorHoles":interiorHolesName}

        self.simplifyPolygons = simplifyPolygons
        
        self.setSimplifyTolerance(meshMaxSize**0.5)


        self.meshMaxSize = meshMaxSize #m2
        self.defaultBcTagSuffix = "bd"    # defualt suffix for the boundary tag
        
        # variables assign in functions
        self.domain = None
        self.inflow = None
        self.tagsDict = {}

        pass

    def setSimplifyTolerance(self,tol:float):
        '''
        tol = tolaerance (m) e.g. meshMaxSize**0.5
        '''
        self.simplifyTolerance = tol

    def createDomain(self):
        '''
        works on the domian
        1) reads all geo files, settings in self.fileNamesDict
        2) creates indexes and tags for Boundaries conditions (user supplied + default)
            ->self.tagsDict
        3) Reads DTM (by def. converts it to ESRI asc, and cleans files)
            self.domain

        '''
        # read boundary
        boundary = featuresToPolylines(self.settingsDict["Boundaries"])[0]

        if self.simplifyPolygons:
            # simplify polygon mesh, avoiding to coarse mesh
            boundary = simplify_polygon_coords(boundary, self.simplifyTolerance)
            

        # read user specified BC tags
        if self.settingsDict["boundaryConditionsZonesGeoName"] is not None:
            BcsPol, BcTags = featuresToPolylines(
                featureFile=self.settingsDict["boundaryConditionsZonesGeoName"],
                field=self.settingsDict["boundaryConditionsTagsField"])
            
            # Adding points at poligon intersection

            #BcIntersections:
            BcIntersections = []

            for bc in BcsPol:
                c = find_vector_intersection(boundary, bc)
                BcIntersections.extend (c)


            updatedBoundary, indexes = includePointsToVector(boundary,BcIntersections)


            # indeksy punktów dla warunków brzegowych


            bcIndexes = get_bc_masks(BcsPol, updatedBoundary,buffer_distance=1.)


        else:
            BcTags = []
            bcIndexes = []
            updatedBoundary=boundary
        

        defTagSuffix = self.defaultBcTagSuffix


        # self.tagsDict = {}

        defTagCounter = -1
        defaultFlag = False
        

        # tag generator
        # loop over updatedBoundary, if within bcIndexes -- apply BcTags, else apply using given def tag name

        # ok mam indeksy gdzie wpada warunek brzegowy.
        # indeksy odcinków do indeksy punktów +1

        for k in range(len(updatedBoundary)):
            tag = ""
            # Checking if within registered tags:
            for inds,t in zip(bcIndexes,BcTags):
                i0,i1 = inds
                if k >= i0 and k< i1:
                    tag = t
                    defaultFlag = False
                    break

            if tag == "": # no tag found, giving default name
                # checking if last key wasn't also default one:
                if not defaultFlag:
                    defaultFlag = True
                    defTagCounter +=1

                tag = f"{defTagSuffix}_{defTagCounter}"
            
            if tag in self.tagsDict:
                # adding index to existing entry:
                self.tagsDict[tag].append(k)
            else:
                self.tagsDict[tag] = [k]

        
        # Read holes:
        if self.settingsDict["interiorHoles"] is not None:
            holePolygons = featuresToPolylines(self.settingsDict["interiorHoles"])
            
            # def bc tag: "interior"
            # holse should be in CCW order:
            for hole in holePolygons:
                if signedArea(hole) <0:
                    hole.reverse()
            if self.simplifyPolygons:
                for i in range(len(holePolygons)):
                    holePolygons[i] = simplify_polygon_coords(holePolygons[i],self.simplifyTolerance)


        else:
            holePolygons = None


        # create domain

        self.domain = anuga.create_domain_from_regions(
            updatedBoundary,
            boundary_tags=self.tagsDict,
            maximum_triangle_area=self.meshMaxSize,
            interior_holes = holePolygons)

        self.domain.set_name(self.settingsDict["domainName"]) # Name of sww file
        
        # read dtm




        ## DTM   
        self._readQuantityFromRaster(self.settingsDict["DTM"],quantityName="elevation")
        
        ## roughness

        if type(self.settingsDict["roughness"]) is float:
            self.domain.set_quantity("friction",self.settingsDict["roughness"])
        else:
            # raster
            self._readQuantityFromRaster(self.settingsDict["roughness"],quantityName="friction")
        
        
        # initial stage
        self.domain.set_quantity("stage",self.settingsDict["initialStage"])
        
        pass


    def _isRasterAsc(self,rasterName:str) -> bool:
        '''
        To improve
        '''
        if rasterName[-3:] == "asc":
            return True
        else:
            return False


    def _readQuantityFromRaster(self,rasterName:str,quantityName:str):

        ## Raster -- conversion to asc

        # is asc?

        if self._isRasterAsc(rasterName):
            self.domain.set_quantity(quantityName, filename=rasterName, location='centroids') # Use function for elevation
        else:

            # creating random file for the ESRI dtm
            randomNumber = np.random.randint(0,1000)

            temporaryFile = f"tmpRST_{randomNumber:04.0f}.asc"

            try:
                # Convert the DTM to ASC
                convert_dtm_to_asc(rasterName, temporaryFile)

                # Process the ASC file
                self.domain.set_quantity(quantityName, filename=temporaryFile, location='centroids') # Use function for elevation



            finally:
                # Clean up: Delete the ASC file
                if os.path.exists(temporaryFile):
                    try:
                        os.remove(temporaryFile)
                        
                    except OSError as e:
                        print(f"Error deleting {temporaryFile}: {e}")

    def setBoundaryConditions(self,inflowTag:str,outflowTag:str,outflowVal:float):

        def outflowStage(t):
            return  outflowVal 

        Bo = anuga.Transmissive_momentum_set_stage_boundary(self.domain, 
                                                            function=outflowStage)         # Outflow

        Br = anuga.Reflective_boundary(self.domain)            # Solid reflective wall
        # Bt = anuga.Transmissive_boundary(self.domain)


        # remaining boundaries
        tabBd = {inflowTag:Br, outflowTag:Bo}
        for tag in self.domain.get_boundary_tags():
            if tag not in tabBd:
                tabBd[tag] = Br

        # checking for interior holes:
        if self.settingsDict["interiorHoles"] is not None:
            tabBd['interior'] = Br


        self.domain.set_boundary(tabBd)


        pass
        # anuga

    def setInflowLine(self,Q:float):
                # read inflow poly
        inflowLine = featuresToPolylines(self.settingsDict["InletLine"])[0]

        inflowZone = anuga.Region(self.domain, line = inflowLine )

        self.inflow = anuga.Inlet_operator(self.domain, inflowZone , Q=Q)
        pass


