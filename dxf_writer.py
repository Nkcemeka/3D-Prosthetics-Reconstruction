"""
Project Author: Chukwuemeka L. Nkama
Date: July 21, 2023

dxf_writer.py is a file that takes a csv file containing
coordinates of a contour and uses that to generate a dxf
file which can be opened in softwares like FreeCAD, Auto-
CAD etc...
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import sys
import shapely
import geopandas as gpd
import ezdxf


# Read input file
if len(sys.argv) == 3:
    csv_path = sys.argv[1]
    output_path = sys.argv[2]
else:
    print("ERROR: Format is python3 dxf_writer.py /path/to/database \
            output_filename")
    sys.exit()

database = pd.read_csv(csv_path)

# Create GeoPoints
zipped_points = zip(database['x'], database['y'])
geoPoints = [shapely.geometry.Point(point) for point in zipped_points]

# Add csv data and GeoPoints to central database
database['GeoPoints'] = geoPoints

# Get GeoLines
lines = []
for pos in range(len(database)-1):
    geoline = shapely.geometry.LineString([database['GeoPoints'][pos], \
            database['GeoPoints'][pos+1]])
    lines.append(geoline)

# Add the last line to make the contour closed
lines.append(shapely.geometry.LineString([database['GeoPoints'][len(database)-1], \
        database['GeoPoints'][0]]))

# Create database holding lines
db_lines = pd.DataFrame({'GeoLines':lines})

# Create file and model space
doc = ezdxf.new(setup=True) # setup .dxf file 
msp = doc.modelspace()

# Create a layer for the lines
lines_layer = doc.layers.add("GeoLines")
lines_layer.color = 5 # blue color

# Add the GeoLines to the ModelSpace
for pos in range(len(db_lines)):
    msp.add_lwpolyline(db_lines["GeoLines"][pos].coords, dxfattribs={"layer":"GeoLines"})

# Save the dxf file
doc.saveas(f"{output_path}.dxf")
