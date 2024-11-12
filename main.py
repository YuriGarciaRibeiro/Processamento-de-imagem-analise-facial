import cv2

img = cv2.imread('image.webp')

detec_face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

detec_eye = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = detec_face.detectMultiScale(cinza, 1.3, 3)

eye = detec_eye.detectMultiScale(cinza, 1.197, 3)

for (x, y, larg, alt) in eye:  # Desenhar o retângulo
    ret = cv2.rectangle(img, (x, y), (x + larg, y + alt), (255, 0, 0), 3)

for (x, y, larg, alt) in face:  # Desenhar o retângulo
    ret = cv2.rectangle(img, (x, y), (x + larg, y + alt), (0, 255, 0), 3)

cv2.imshow("", img)
cv2.waitKey(0)
