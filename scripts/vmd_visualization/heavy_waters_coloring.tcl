# load PDB with bfactors containing number of heavy waters 6 A around
# each heavy atom
# color according to this at each timestep
#
# @author: Akash Pallath
# Uses pdbbfactor (http://www.ks.uiuc.edu/Research/vmd/script_library/scripts/pdbbfactor/)

source pdbbfactor.tcl

#load values from beta -> user (which is set per timestep)
#SET PDB FILE NAME HERE
pdbbfactor <>

#Go to first frame
animate goto start

#delete default display
mol delrep 0 top

#use VDW representation
mol representation VDW 0.700000 12.000000

#color by User column
mol color User

#only visualize protein heavy atoms
mol selection {protein and not name "[0-9]?H.*" }

#use Opaque materials
mol material Diffuse

#switch off depth aliasing
display depthcue on

#add this representation
mol addrep top

#remove axes
axes location Off

#set display
display projection Orthographic
display resize 800 800
scale to 0.025
color Display background white

#show color bar menu
menu colorscalebar on
