import cv2

def nothing(x):
    pass

scale_factor_face = 1.3
min_neighbors_face = 5
scale_factor_eye = 1.31
min_neighbors_eye = 2
scale_factor_smile = 1.204
min_neighbors_smile = 30

detec_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detec_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
detec_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

cv2.namedWindow("Ajustes")

cv2.createTrackbar("Scale Face", "Ajustes", int(scale_factor_face * 10), 30, nothing)
cv2.createTrackbar("Min Neighbors Face", "Ajustes", min_neighbors_face, 20, nothing)

cv2.createTrackbar("Scale Eye", "Ajustes", int(scale_factor_eye * 10), 30, nothing)
cv2.createTrackbar("Min Neighbors Eye", "Ajustes", min_neighbors_eye, 20, nothing)

cv2.createTrackbar("Scale Smile", "Ajustes", int(scale_factor_smile * 10), 30, nothing)
cv2.createTrackbar("Min Neighbors Smile", "Ajustes", min_neighbors_smile, 50, nothing)

while True:
    ret, img = cap.read()
    if not ret:
        break

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_factor_face = max(cv2.getTrackbarPos("Scale Face", "Ajustes") / 10, 1.1)
    min_neighbors_face = cv2.getTrackbarPos("Min Neighbors Face", "Ajustes")

    scale_factor_eye = max(cv2.getTrackbarPos("Scale Eye", "Ajustes") / 10, 1.1)
    min_neighbors_eye = cv2.getTrackbarPos("Min Neighbors Eye", "Ajustes")

    scale_factor_smile = max(cv2.getTrackbarPos("Scale Smile", "Ajustes") / 10, 1.1)
    min_neighbors_smile = cv2.getTrackbarPos("Min Neighbors Smile", "Ajustes")

    faces = detec_face.detectMultiScale(
        cinza, scaleFactor=scale_factor_face, minNeighbors=min_neighbors_face)

    for (x, y, larg, alt) in faces:
        cv2.rectangle(img, (x, y), (x + larg, y + alt), (0, 255, 0), 3)

        roi_gray = cinza[y:y + alt, x:x + larg]
        roi_color = img[y:y + alt, x:x + larg]

        eyes = detec_eye.detectMultiScale(
            roi_gray, scaleFactor=scale_factor_eye, minNeighbors=min_neighbors_eye)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),(ex + ew, ey + eh), (255, 0, 0), 2)
            print("Olhos O_O")

        smiles = detec_smile.detectMultiScale(
            roi_gray, scaleFactor=scale_factor_smile, minNeighbors=min_neighbors_smile)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            print("Sorriso :D")

    cv2.imshow("Deteccãao de Rosto, Olhos e Sorrisos", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
