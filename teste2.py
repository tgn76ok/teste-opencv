import cv2
  
img = cv2.imread("resource/carro3.jpg")  # Read image

  
# Defining all the parameters
t_lower = 100 # Lower Threshold
t_upper = 200 # Upper threshold
aperture_size = 3 # Aperture size
L2Gradient = True # Boolean
  
# Applying the Canny Edge filter 
# with Aperture Size and L2Gradient
edge = cv2.Canny(img, t_lower, t_upper,
                 apertureSize = aperture_size, 
                 L2gradient = L2Gradient ) 
  
cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()