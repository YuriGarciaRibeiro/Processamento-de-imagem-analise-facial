import cv2

# Carrega a imagem
img = cv2.imread('image.webp')
print(cv2.data.haarcascades)
# Carrega os classificadores Haar para rosto e olhos
detec_face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detec_eye = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
detec_smile = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

# Converte a imagem para escala de cinza
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecta rostos na imagem
faces = detec_face.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)

# Loop para cada rosto detectado
for (x, y, larg, alt) in faces:
    # Desenha um retângulo ao redor do rosto
    cv2.rectangle(img, (x, y), (x + larg, y + alt), (0, 255, 0), 3)

    # Define a região de interesse (ROI) para os olhos dentro do rosto
    roi_gray = cinza[y:y + alt, x:x + larg]
    roi_color = img[y:y + alt, x:x + larg]

    # Detecta olhos dentro da ROI do rosto
    eyes = detec_eye.detectMultiScale(
        roi_gray, scaleFactor=1.1, minNeighbors=2)

    # Desenha retângulos ao redor dos olhos
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # Detecta sorrisos dentro da ROI do rosto
    smiles = detec_smile.detectMultiScale(
        roi_gray, scaleFactor=1.204, minNeighbors=30)

    # Desenha retângulos ao redor dos sorrisos
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)


# Exibe a imagem com os retângulos
cv2.imshow("Detecção de Rosto e Olhos", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
