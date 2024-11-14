import cv2

img = cv2.imread('./images/image.webp')
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scale_factor_face = 1.3
min_neighbors_face = 5
scale_factor_eye = 1.31
min_neighbors_eye = 2
scale_factor_smile = 1.204
min_neighbors_smile = 30

detec_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detec_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
detec_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

faces = detec_face.detectMultiScale(
    cinza, scaleFactor=scale_factor_face, minNeighbors=min_neighbors_face)

for (x, y, larg, alt) in faces:
    cv2.rectangle(img, (x, y), (x + larg, y + alt), (0, 255, 0), 3)

    roi_gray = cinza[y:y + alt, x:x + larg]
    roi_color = img[y:y + alt, x:x + larg]

    eyes = detec_eye.detectMultiScale(
        roi_gray, scaleFactor=scale_factor_eye, minNeighbors=min_neighbors_eye)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    smiles = detec_smile.detectMultiScale(
        roi_gray, scaleFactor=scale_factor_smile, minNeighbors=min_neighbors_smile)

    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

cv2.imshow("Deteccao de Rosto e Olhos", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
