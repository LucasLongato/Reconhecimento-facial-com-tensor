import cv2
import imutils.paths as paths
import face_recognition
import pickle
import os

dataset = "DATASET"  # caminho do conjunto de dados
module = "treinamento.pickle"  # caminho onde você deseja armazenar o arquivo pickle

image_paths = list(paths.list_images(dataset))
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
data = {"Treinamento": known_encodings, "nomes": known_names}
output = open(module, "wb")
pickle.dump(data, output)
output.close()

print("Treinamento salvos em {}".format(module))
