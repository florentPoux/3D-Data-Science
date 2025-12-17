"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 10

General Information:
-------------------
* 🦊 Created by:    Florent Poux
* 📅 Last Update:   Dec. 2025
* © Copyright:      Florent Poux
* 📜 License:       MIT

Dependencies:
------------
* Environment:      Anaconda or Miniconda
* Python Version:   3.9+
* Key Libraries:    NumPy, Pandas, Open3D, Laspy, Scikit-Learn

Helpful Links:
-------------
* 🏠 Author Website:        https://learngeodata.eu
* 📚 O'Reilly Book Page:    https://www.oreilly.com/library/view/3d-data-science/9781098161323/

Enjoy this code! 🚀
"""
# -*- coding: utf-8 -*-
"""
General Information
* Created by: 🦊 Florent Poux. 
* Copyright: Florent Poux.
* License: MIT
* Status: Online

Dependencies:
* Anaconda or Miniconda
* An Anaconda new environment
* Libraries as described in the Chapter

Have fun with this Code Solution.

🎵 Note: Styling was not taken care of at this stage.

Enjoy!
"""

import cadquery as cq
import open3d as o3d
import numpy as np

#%% 1. Basic Modelling Techniques

# https://cadquery.readthedocs.io/en/latest/examples.html#plate-with-hole

cube = cq.Workplane().box(1, 1, 1)

# Show the cube
cq.show(cube)

#%% 1.1. Sketching (2D CAD)

pts = [(0,0),(0,20),(12,20),(12,18),(2,18),(2,16),(8,16),(8,14),(2,14),(2,0)]

flo = cq.Workplane("front").polyline(pts).close()

cq.exporters.export(flo, "../RESULTS/letter.dxf")

# Example better

(L, H, W, t) = (100.0, 20.0, 20.0, 1.0)
pts = [
    (0, H / 2.0),
    (W / 2.0, H / 2.0),
    (W / 2.0, (H / 2.0 - t)),
    (t / 2.0, (H / 2.0 - t)),
    (t / 2.0, (t - H / 2.0)),
    (W / 2.0, (t - H / 2.0)),
    (W / 2.0, H / -2.0),
    (0, H / -2.0),
]
result = cq.Workplane("front").polyline(pts).mirrorY().extrude(L)
cq.exporters.export(result, "../RESULTS/IPN.stl")

#%% 1.2. Extruding

flo_e = cq.Workplane("front").polyline(pts).close().extrude(1)
cq.exporters.export(flo_e, "../RESULTS/letter.stl")

#%% Visualize within Python

def visualize_object(cq_object):
    temp_path = "../RESULTS/temp.stl"
    cq.exporters.export(cq_object, temp_path)
    o3d_mesh = o3d.io.read_triangle_mesh(temp_path)
    o3d_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([o3d_mesh])

#%% Revolving

# Define the profile
profile = cq.Sketch()
with profile:
    profile.rect(1, 2).finalize()

# Extrude the profile
extruded = profile.extrude(2)

# Revolve the extruded shape
result = extruded.revolve(360)
visualize_object(result)

#%% 1.3 Fillet

result = cq.Workplane("XY").box(3, 3, 0.5)
visualize_object(result)

result = cq.Workplane("XY").box(3, 3, 0.5).edges("|Z").fillet(0.125)
visualize_object(result)

result = cq.Workplane("XY").box(3, 3, 0.5).faces("+Z").chamfer(0.1)
visualize_object(result)

#%% 1.4 Mirorring

r = cq.Workplane("front").hLine(1.0)  # 1.0 is the distance, not coordinate

r = (
    r.vLine(0.5).hLine(-0.25).vLine(-0.25).hLineTo(0.0)
)  # hLineTo allows using xCoordinate not distance

result = r.mirrorY().extrude(0.25)  # mirror the geometry and extrude
visualize_object(result)

#%% 2. Boolean Operations

# The dimensions of the box. These can be modified rather than changing the
# object's code directly.
length = 80.0
height = 60.0
thickness = 10.0
center_hole_dia = 22.0

# Create a box based on the dimensions above and add a 22mm center hole
cube_element = (
    cq.Workplane("XY")
    .box(length, height, thickness)
    .faces(">Z")
    .workplane()
    .hole(center_hole_dia)
)

#Temp File:
result_path = "../RESULTS/sample_mesh.stl"

# Display the cube with Open3D
# Convert the CadQuery solid to a mesh for Open3D compatibility
cq.exporters.export(cube_element, result_path)
#mesh = cq.mesh(cube_element)

# Create an Open3D mesh object from the CadQuery mesh data
# open3d_mesh = o3d.geometry.TriangleMesh(mesh.points, mesh.triangles)
o3d_mesh = o3d.io.read_triangle_mesh(result_path)
o3d_mesh.compute_vertex_normals()

# Visualize the mesh with Open3D
o3d.visualization.draw_geometries([o3d_mesh])

#%%

def visualize_object(cq_object):
    temp_path = "../RESULTS/temp.stl"
    cq.exporters.export(cq_object, temp_path)
    o3d_mesh = o3d.io.read_triangle_mesh(temp_path)
    o3d_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([o3d_mesh])

#%% Boolean Opeations Example

Box = cq.Workplane("XY").box(1, 1, 1, centered=(False, False, False))
Sphere = cq.Workplane("XY").sphere(1)

#%% CSG: Intersection
result = Box & Sphere
visualize_object(result)

# CSG: Union
result = Box | Sphere
visualize_object(result)

# CSG: Difference
result = Box - Sphere
visualize_object(result)

#%% Modelling objects:
    
#%% IPN

(L, H, W, t) = (100.0, 20.0, 20.0, 1.0)
pts = [
    (0, H / 2.0),
    (W / 2.0, H / 2.0),
    (W / 2.0, (H / 2.0 - t)),
    (t / 2.0, (H / 2.0 - t)),
    (t / 2.0, (t - H / 2.0)),
    (W / 2.0, (t - H / 2.0)),
    (W / 2.0, H / -2.0),
    (0, H / -2.0),
]
result = cq.Workplane("front").polyline(pts).mirrorY().extrude(L)
visualize_object(result)

#%% Spherical joint

sphere = cq.Workplane().sphere(5)
base = cq.Workplane(origin=(0, 0, -2)).box(12, 12, 10).cut(sphere).edges("|Z").fillet(2)
sphere_face = base.faces(">>X[2] and (not |Z) and (not |Y)").val()
base = base.faces("<Z").workplane().circle(2).extrude(10)

shaft = cq.Workplane().sphere(4.5).circle(1.5).extrude(20)

spherical_joint = (
    base.union(shaft)
    .faces(">X")
    .workplane(centerOption="CenterOfMass")
    .move(0, 4)
    .slot2D(10, 2, 90)
    .cutBlind(sphere_face)
    .workplane(offset=10)
    .move(0, 2)
    .circle(0.9)
    .extrude("next")
)

result = spherical_joint
visualize_object(result)

#%% Parametric enclosure

# parameter definitions
p_outerWidth = 100.0  # Outer width of box enclosure
p_outerLength = 150.0  # Outer length of box enclosure
p_outerHeight = 50.0  # Outer height of box enclosure

p_thickness = 3.0  # Thickness of the box walls
p_sideRadius = 10.0  # Radius for the curves around the sides of the box
p_topAndBottomRadius = (
    2.0  # Radius for the curves on the top and bottom edges of the box
)

p_screwpostInset = 12.0  # How far in from the edges the screw posts should be place.
p_screwpostID = 4.0  # Inner Diameter of the screw post holes, should be roughly screw diameter not including threads
p_screwpostOD = 10.0  # Outer Diameter of the screw posts.\nDetermines overall thickness of the posts

p_boreDiameter = 8.0  # Diameter of the counterbore hole, if any
p_boreDepth = 1.0  # Depth of the counterbore hole, if
p_countersinkDiameter = 0.0  # Outer diameter of countersink. Should roughly match the outer diameter of the screw head
p_countersinkAngle = 90.0  # Countersink angle (complete angle between opposite sides, not from center to one side)
p_flipLid = True  # Whether to place the lid with the top facing down or not.
p_lipHeight = 1.0  # Height of lip on the underside of the lid.\nSits inside the box body for a snug fit.

# outer shell
oshell = (
    cq.Workplane("XY")
    .rect(p_outerWidth, p_outerLength)
    .extrude(p_outerHeight + p_lipHeight)
)

# weird geometry happens if we make the fillets in the wrong order
if p_sideRadius > p_topAndBottomRadius:
    oshell = oshell.edges("|Z").fillet(p_sideRadius)
    oshell = oshell.edges("#Z").fillet(p_topAndBottomRadius)
else:
    oshell = oshell.edges("#Z").fillet(p_topAndBottomRadius)
    oshell = oshell.edges("|Z").fillet(p_sideRadius)

# inner shell
ishell = (
    oshell.faces("<Z")
    .workplane(p_thickness, True)
    .rect((p_outerWidth - 2.0 * p_thickness), (p_outerLength - 2.0 * p_thickness))
    .extrude(
        (p_outerHeight - 2.0 * p_thickness), False
    )  # set combine false to produce just the new boss
)
ishell = ishell.edges("|Z").fillet(p_sideRadius - p_thickness)

# make the box outer box
box = oshell.cut(ishell)

# make the screw posts
POSTWIDTH = p_outerWidth - 2.0 * p_screwpostInset
POSTLENGTH = p_outerLength - 2.0 * p_screwpostInset

box = (
    box.faces(">Z")
    .workplane(-p_thickness)
    .rect(POSTWIDTH, POSTLENGTH, forConstruction=True)
    .vertices()
    .circle(p_screwpostOD / 2.0)
    .circle(p_screwpostID / 2.0)
    .extrude(-1.0 * (p_outerHeight + p_lipHeight - p_thickness), True)
)

# split lid into top and bottom parts
(lid, bottom) = (
    box.faces(">Z")
    .workplane(-p_thickness - p_lipHeight)
    .split(keepTop=True, keepBottom=True)
    .all()
)  # splits into two solids

# translate the lid, and subtract the bottom from it to produce the lid inset
lowerLid = lid.translate((0, 0, -p_lipHeight))
cutlip = lowerLid.cut(bottom).translate(
    (p_outerWidth + p_thickness, 0, p_thickness - p_outerHeight + p_lipHeight)
)

# compute centers for screw holes
topOfLidCenters = (
    cutlip.faces(">Z")
    .workplane(centerOption="CenterOfMass")
    .rect(POSTWIDTH, POSTLENGTH, forConstruction=True)
    .vertices()
)

# add holes of the desired type
if p_boreDiameter > 0 and p_boreDepth > 0:
    topOfLid = topOfLidCenters.cboreHole(
        p_screwpostID, p_boreDiameter, p_boreDepth, 2.0 * p_thickness
    )
elif p_countersinkDiameter > 0 and p_countersinkAngle > 0:
    topOfLid = topOfLidCenters.cskHole(
        p_screwpostID, p_countersinkDiameter, p_countersinkAngle, 2.0 * p_thickness
    )
else:
    topOfLid = topOfLidCenters.hole(p_screwpostID, 2.0 * p_thickness)

# flip lid upside down if desired
if p_flipLid:
    topOfLid = topOfLid.rotateAboutCenter((1, 0, 0), 180)

# return the combined result
result = topOfLid.union(bottom)
visualize_object(result)

#%% Cycloidal gear

import cadquery as cq
from math import sin, cos, pi, floor


# define the generating function
def hypocycloid(t, r1, r2):
    return (
        (r1 - r2) * cos(t) + r2 * cos(r1 / r2 * t - t),
        (r1 - r2) * sin(t) + r2 * sin(-(r1 / r2 * t - t)),
    )


def epicycloid(t, r1, r2):
    return (
        (r1 + r2) * cos(t) - r2 * cos(r1 / r2 * t + t),
        (r1 + r2) * sin(t) - r2 * sin(r1 / r2 * t + t),
    )


def gear(t, r1=4, r2=1):
    if (-1) ** (1 + floor(t / 2 / pi * (r1 / r2))) < 0:
        return epicycloid(t, r1, r2)
    else:
        return hypocycloid(t, r1, r2)


# create the gear profile and extrude it
result = (
    cq.Workplane("XY")
    .parametricCurve(lambda t: gear(t * 2 * pi, 6, 1))
    .twistExtrude(15, 90)
    .faces(">Z")
    .workplane()
    .circle(2)
    .cutThruAll()
)

visualize_object(result)

#%% Combining Walls

# Define parameters (modify these to customize the house)
length = 5  # Length of the house in m
width = 3  # Width of the house in m
height = 2.5  # Height of the house in m

wall_thickness = 0.2

walls = cq.Workplane("XY").box(length, wall_thickness, height)

w1 = walls
w2 = walls.translate((0, width, 0))
w3 = cq.Workplane("XY").box(width + wall_thickness, wall_thickness, height).rotate((0, 0, 0), (0, 0, 1), 90).translate((length/2, width/2, 0))
w4 = cq.Workplane("XY").box(width + wall_thickness, wall_thickness, height).rotate((0, 0, 0), (0, 0, 1), 90).translate((-length/2, width/2, 0))

result = w1 | w2 | w3 | w4
visualize_object(result)

#%% Creating Doors
# object = L,W,H,Cx,Cy,Cz,
d_par = np.array([1, 0.2, 2])
door_relative_shift = np.array([-1, 0, -0.25])

d1 = cq.Workplane("XY").box(d_par[0], d_par[1], d_par[2])
d1_t = cq.Workplane("XY").box(d_par[0], d_par[1], d_par[2], centered=True).translate((list(door_relative_shift)))

visualize_object(d1 | d1_t)

result_d = result - d1_t
visualize_object(result_d)


#%% Generating the ground Ground

ground = cq.Workplane("XY").box(length+0.2, width+0.2, 0.1).translate((0,width/2,-height/2-0.1))

result_total = result_d | ground
visualize_object(result_total)

#%%

# create 4 small square bumps on a larger base plate:
s = (
    cq.Workplane()
    .box(4, 4, 0.5)
    .faces(">Z")
    .workplane()
    .rect(3, 3, forConstruction=True)
    .vertices()
    .box(0.25, 0.25, 0.25, combine=True)
)

visualize_object(s)

#%%

# Define parameters (modify these to customize the house)
length = 5  # Length of the house in m
width = 3  # Width of the house in m
height = 2.5  # Height of the house in m

wall_thickness = 0.2

door_width = 1  # Width of the door in m
door_height = 4  # Height of the door in m
window_width = 1.5  # Width of the window in m
window_height = 1  # Height of the window in m
roof_pitch = 45  # Roof pitch angle in degrees

result = cq.Workplane("front").rect(length+wall_thickness, width+wall_thickness).rect(length, width).extrude(height)
visualize_object(result)



#%%

# s = cq.Sketch().rect(door_width, door_height).vertices().fillet(0.02)
s = cq.Sketch().rect(door_width, door_height)

result2 = (
    cq.Workplane("front")
    .rect(length+wall_thickness, width+wall_thickness)
    .rect(length, width)
    .extrude(height)
    .faces(">X")
    .workplane()
    #.transformed((0, 0, -90))
    .placeSketch(s)
    .cutThruAll()
)

visualize_object(result2)

#%%

ground = cq.Workplane("XY").box(length+0.2, width+0.2, 0.1)

result_total = result2 | ground
visualize_object(result_total)