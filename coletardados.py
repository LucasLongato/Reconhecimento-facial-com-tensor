import numpy as np
import os
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
path = ''  # Caminho onde você deseja armazenar o conjunto de dados

while True:
    nome = input('Digite o nome do usuário (ou digite "sair" para encerrar): ')
    
    if nome.lower() == 'sair':
        break

    # Cria a pasta se ainda não existir
    os.makedirs(os.path.join('DATASET', nome), exist_ok=True)

    sampleN = 0

    while True:
        ret, img = cap.read()
        frame = img.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleN += 1
            cv2.imwrite(os.path.join('DATASET', nome, str(sampleN) + '.jpg'), gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.waitKey(100)

        cv2.imshow('Reconhecimento da face', img)
        cv2.waitKey(1)

        if sampleN > 100:
            break

    print(f'Conjunto de dados salvo para {nome}.')

cap.release()
cv2.destroyAllWindows()

