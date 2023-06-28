import numpy as np
import os
import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
dataset_path = 'DATASET'  # Caminho onde você deseja armazenar o conjunto de dados

def capture_dataset():
    while True:
        nome = input('Digite o nome do usuário (ou digite "sair" para encerrar): ')
        
        if nome.lower() == 'sair':
            break

        # Cria a pasta se ainda não existir
        os.makedirs(os.path.join(dataset_path, nome), exist_ok=True)

        sampleN = 0
        start_time = time.time()

        while True:
            ret, img = cap.read()
            frame = img.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sampleN += 1
                cv2.imwrite(os.path.join(dataset_path, nome, str(sampleN) + '.jpg'), gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.waitKey(100)

            cv2.imshow('Reconhecimento da face', img)
            cv2.waitKey(1)

            if sampleN > 100 or time.time() - start_time > 30:
                break

        print(f'Conjunto de dados salvo para {nome}.')

def train_dataset():
    import imutils.paths as paths
    import face_recognition
    import pickle

    image_paths = list(paths.list_images(dataset_path))
    known_encodings = []
    known_names = []

    for (i, image_path) in enumerate(image_paths):
        print("[INFO] processando imagem {}/{}".format(i + 1, len(image_paths)))
        name = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    print("[INFO] serializando os encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    output = open("treinamento.pickle", "wb")
    pickle.dump(data, output)
    output.close()

    print("Treinamento salvo em treinamento.pickle")

def recognize_faces():
    import imutils
    import pickle
    import face_recognition

    encoding = "treinamento.pickle"
    data = pickle.loads(open(encoding, "rb").read())

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Falha ao capturar o quadro da câmera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=400)
        r = frame.shape[1] / float(rgb.shape[1])

        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        if len(boxes) == 0:
            elapsed_time = 0
            start_time = time.time()
            while elapsed_time < 30:
                cv2.imshow("Frame", frame)
                elapsed_time = time.time() - start_time
                if cv2.waitKey(1) == 27:
                    break
            if elapsed_time >= 30:
                nome = input('Pessoa não reconhecida. Digite o nome do usuário: ')
                if nome.lower() == 'sair':
                    break
                os.makedirs(os.path.join(dataset_path, nome), exist_ok=True)
                sampleN = 0
                start_time = time.time()
                while True:
                    ret, img = cap.read()
                    frame = img.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        sampleN += 1
                        cv2.imwrite(os.path.join(dataset_path, nome, str(sampleN) + '.jpg'), gray[y:y+h, x:x+w])
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.waitKey(100)
                    cv2.imshow('Reconhecimento da face', img)
                    cv2.waitKey(1)
                    if sampleN > 100 or time.time() - start_time > 30:
                        break
                print(f'Conjunto de dados salvo para {nome}.')
            continue

        for encoding in encodings:
            matches = face_recognition.compare_faces(np.array(encoding), np.array(data["encodings"]))
            name = "Desconhecido"
               
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                   
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
            names.append(name)
                
        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r) 
            left = int(left * r)
            
            if name == "Desconhecido":
                # Desenha o retângulo e o texto com "Desconhecido"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                # Desenha o retângulo e o texto com o nome reconhecido
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Imprime o nome da pessoa reconhecida no terminal
            print("Pessoa reconhecida:", name)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def main():
    capture_dataset()
    train_dataset()
    recognize_faces()

if __name__ == "__main__":
    main()
