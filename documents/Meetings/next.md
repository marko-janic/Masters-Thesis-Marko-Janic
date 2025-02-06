- Right now to "identify" the sub micrographs I also with each micrograph give the coordinates of the top left most 
point in the image, is this a good way to do it or would you recommend something else?
- When I calculate the loss, should I do so for the coordinates in the sub_micrograph or the coordinates in the 
original image given the sub micrograph
- Currently im "hard coding" the box sizes (width/heigh) in the loss funciton, I still need to fix this so it just uses
the center coordintaes. Do you think it makes sense for me to keep it in case we want to do the box loss stuff?