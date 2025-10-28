
"""Run parallel shallow water domain.

   run using command like:

   mpiexec -np m --use-hwthread-cpus python test4mpi.py
   mpiexec -np 6 python test4mpi.py

   where m is the number of processors to be used.
   
   Will produce sww files with names domain_Pn_m.sww where n is number of processors and
   m in [0, n-1] refers to specific processor that owned this part of the partitioned mesh.
"""

#------------------------------------------------------------------------------
# Import necessary modules
#------------------------------------------------------------------------------

import os
import sys
import time
import numpy as np

#------------------------
# ANUGA Modules
#------------------------
	
import anuga

from anuga import distribute, myid, numprocs, finalize, barrier





# from anugaTools import extractResultsAlongLine

# my
from createCaseAnuga import createCase

#--------------------------------------------------------------------------
# Setup parameters
#--------------------------------------------------------------------------

verbose = True

#--------------------------------------------------------------------------
# Setup procedures
#--------------------------------------------------------------------------

tStep = 60*60
finalTime = 24*60*60

CellRes = 400 
domainName = f"KozieniceTest_r{CellRes}"

boundaryFile = "shp/anugaModelBoundary.gpkg"
bcFile = "shp/anugaModelBcPolygons.gpkg"
dtmName= "dtm/bathymetry2.tif"
dtmName= 'tmpDTM_0729.asc'
inletFile = "shp/anugaInlet.gpkg"



case = createCase(dtmName, boundaryFile, inletFile,
                            roughnessRstOrValue = 0.025,
                            initialStage=105,
                          meshMaxSize=CellRes,
                          domainName=domainName,
                          boundaryConditionsZonesGeoName = bcFile,
                          boundaryConditionsTagsField = "Tag")





#--------------------------------------------------------------------------
# Setup Domain only on processor 0
#--------------------------------------------------------------------------
if myid == 0:

    case.createDomain()

else:
    pass
    # domain = None #set in case

#--------------------------------------------------------------------------
# Distribute sequential domain on processor 0 to other processors
#--------------------------------------------------------------------------

if myid == 0 and verbose: 
    print ('DISTRIBUTING DOMAIN')

case.domain = distribute(case.domain)

#domain.smooth = False
barrier()
for p in range(numprocs):
    if myid == p:
        print ('Process ID %g' %myid)
        print ('Number of triangles %g ' %case.domain.get_number_of_triangles())
        sys.stdout.flush()
        
    barrier()


#sprawdzić to
case.domain.set_flow_algorithm("DE2")

if myid == 0:
    case.domain.print_algorithm_parameters()
    sys.stdout.flush()
    
barrier()

# domain.set_name('domain4mpi')
case.domain.set_store(True)

#------------------------------------------------------------------------------
# Setup boundary conditions
# This must currently happen *after* domain has been distributed
#------------------------------------------------------------------------------

case.setBoundaryConditions(inflowTag ="Wlot",outflowTag="Wylot",outflowVal=103.04)
case.setInflowLine(Q=600)




"""
barrier()
for p in range(numprocs):
    if myid == p:
        print domain.boundary_statistics()
        sys.stdout.flush()
        
    barrier()
"""


#domain.dump_triangulation()

#------------------------------------------------------------------------------
# Evolution
#------------------------------------------------------------------------------
if myid == 0 and verbose: 
    print ('EVOLVE')

t0 = time.time()
# finaltime = 0.25
# yieldstep = 0.05


for t in case.domain.evolve(yieldstep = tStep, finaltime = finalTime):
    if myid == 0:
        case.domain.write_time()



## Profiling
#import cProfile
#prof_file = 'evolve-prof'+ str(numprocs) + '_' + str(myid) + '.dat'
#cProfile.run(s,prof_file)


barrier()

#for id in range(numprocs):
#    if myid == id:
#        import pstats
#        p = pstats.Stats(prof_file)
#        #p.sort_stats('cumulative').print_stats(25)
#        p.sort_stats('time').print_stats(25)
#        sys.stdout.flush
#
#    barrier()
#
#barrier()

for p in range(numprocs):
    if myid == p:
        print ('Process ID %g' %myid)
        print ('Number of processors %g ' %numprocs)
        print ('That took %.2f seconds' %(time.time()-t0))
        print ('Communication time %.2f seconds'%case.domain.communication_time)
        print ('Reduction Communication time %.2f seconds'%case.domain.communication_reduce_time)
        print ('Broadcast time %.2f seconds'%case.domain.communication_broadcast_time)

    barrier()


case.domain.sww_merge(delete_old=True)


if myid==0:
   
   anuga.plot_utils.Make_Geotif(domainName+".sww", 
                     output_quantities=['depth', 'velocity'],
                     myTimeStep='last',
                     CellSize=10.0,
                     EPSG_CODE = 2180,
                     bounding_polygon=case.domain.get_boundary_polygon(),
                     k_nearest_neighbours=1)

finalize()

