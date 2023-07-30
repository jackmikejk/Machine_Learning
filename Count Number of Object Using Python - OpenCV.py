import cv2
import numpy as np
import matplotlib.pyplot as plt

#Reading the image and uploading it in the below variale
uploaded_image = cv2.imread('watershed_coins_01.jpg')

#Converting the uploaded image into graycolor, this conversion is common and plt.imshow & show is to show the converted image
gray_color = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
plt.imshow(gray_color, cmap='gray')
plt.show()

#After converting it into graycolor, the image has been blurred and this method is to reduce the noises in the image
blur_the_image = cv2.GaussianBlur(gray_color, (11,11),0)
plt.imshow(blur_the_image, cmap='gray')
plt.show()

#After blurring the image, the next process is to mark the edges
canny = cv2.Canny(blur_the_image, 30,150,3)
plt.imshow(canny, cmap='gray')
plt.show()

#Even after marking the edges, the lines might be disconnected so the below line is to extend the line so that the lines will be connected
dilated_image = cv2.dilate(canny,(1,1), iterations=0)
plt.imshow(dilated_image, cmap='gray')
plt.show()

(contour, hierarchy) = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
minimum_contour_area = 22
filtered_contour = [cnt for cnt in contour if cv2.contourArea(cnt) > minimum_contour_area]
rgb = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, contour, -1, (0,255,0), 2)
plt.imshow(rgb)
plt.show()


print("Number of Coins in the image is:", len(filtered_contour))















