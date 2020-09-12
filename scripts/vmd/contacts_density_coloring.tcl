# load PDB with bfactors containing number of contacts each atom is part of
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
mol representation VDW 0.500000 12.000000

#color by User column
mol color User

#visualize all protein atoms
#mol selection {protein and not name "[0-9]?H.*" }
mol selection protein

#use Opaque materials
mol material Opaque

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
