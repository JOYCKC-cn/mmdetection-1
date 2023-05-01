import cv2

def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 150)
    return canny

input_image_path = 'path/to/your/image.jpg'
image = cv2.imread(input_image_path)

canny_image = canny_edge(image)

concatenated_image = cv2.hconcat([image, canny_image])

cv2.imshow('Original and Canny Edge Images', concatenated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
