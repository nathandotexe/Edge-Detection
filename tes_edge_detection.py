import cv2
import matplotlib.pyplot as plt

image_path = 'Your Image File Directory'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

edges = cv2.Canny(blurred_image, threshold1=70, threshold2=120)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = 4
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

image_with_contours = image.copy()
cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)

num_objects = len(filtered_contours)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.title("Edges Detected")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.title(f"Contours Detected: {num_objects} objects")
plt.axis("off")

plt.show()
