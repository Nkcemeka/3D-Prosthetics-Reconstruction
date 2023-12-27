# 3D Prosthetics Reconstruction
Rather than trying to rely solely on tradtional 3d reconstrution techniques, this project takes a different approach and aims to see if computer vision methods can be used to obtain relevant measurements from a human residual body part. These measurements would then be used to geenerate a 3D digital twin which would eventually be manipulated with minor changes in order to suit a patient. </br>

## NOTES
1. This project uses deep learning as image processing techniques are too simple to be effective. 
2. It is hard to tell if this method will work but we will see as time goes.
3. Some things are hard-coded into the bash script. So, if you want to change things a bit, go through it.
4. run.sh allows you to pass an optional argument (0 or 1) to flip the orientation of the contour in your CAD software.
5. height.sh allows user to find height between lines

## DXF Files
The code generates a .dxf file that shows the contour of the desired object from a given view. Python script automation in Blender is useful and to see a .dxf file in Blender, some conversion needs to be done. To do this, Blender Add-ons that are suitable can be downloaded and used. Now, although using the raw .dxf file might not be advisable due to the approximations from the segmentation, being able to open it in Blender might help the CAD designer in coming up with his/her relevant 3D model.
