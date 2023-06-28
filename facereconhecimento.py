# import imutils
# import numpy as np
# import pickle
# import cv2
# import face_recognition
# import os
# import time
# import tkinter as tk
# from tkinter import simpledialog
# from imutils import paths
# from treinamentodados import FaceRecognitionTrainer

# def train_dataset(dataset_path, encoding_file):
#     image_paths = list(paths.list_images(dataset_path))
#     known_encodings = []
#     known_names = []

#     for image_path in image_paths:
#         name = image_path.split(os.path.sep)[-2]
#         image = cv2.imread(image_path)
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         boxes = face_recognition.face_locations(rgb, model="hog")
#         encodings = face_recognition.face_encodings(rgb, boxes)

#         for encoding in encodings:
#             known_encodings.append(encoding)
#             known_names.append(name)

#     data = {"encodings": known_encodings, "names": known_names}
#     with open(encoding_file, "wb") as f:
#         pickle.dump(data, f)

#     print("Training completed.")

# def main():
#     encoding_file = "treinamento.pickle"
#     dataset_path = "DATASET"
#     unrecognized_time = 30  # Time in seconds

#     if not os.path.isfile(encoding_file):
#         train_dataset(dataset_path, encoding_file)

#     data = pickle.loads(open(encoding_file, "rb").read())
#     # print(data["key"])

#     cap = cv2.VideoCapture(1)
#     start_time = time.time()
#     unrecognized_timer = 0
#     capture_images = False
#     unknown_person_name = ""

#     if not cap.isOpened():
#         print("Error opening the camera.")
#         return

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             print("Failed to capture frame from camera.")
#             break

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         rgb = imutils.resize(frame, width=400)
#         r = frame.shape[1] / float(rgb.shape[1])

#         boxes = face_recognition.face_locations(rgb, model="hog")
#         encodings = face_recognition.face_encodings(rgb, boxes)
#         names = []

#         for encoding in encodings:
#             matches = face_recognition.compare_faces(np.array(data["Treinamento"]), np.array(encoding))
#             name = "Unknown"

#             if True in matches:
#                 matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#                 counts = {}

#                 for i in matchedIdxs:
#                     name = data["nomes"][i]
#                     counts[name] = counts.get(name, 0) + 1
#                     name = max(counts, key=counts.get)
#             names.append(name)

#         for ((top, right, bottom, left), name) in zip(boxes, names):
#             top = int(top * r)
#             right = int(right * r)
#             bottom = int(bottom * r)
#             left = int(left * r)

#             if name == "Unknown":
#                 if time.time() - start_time >= unrecognized_time:
#                     if not capture_images:
#                         root = tk.Tk()
#                         root.withdraw()
#                         unknown_person_name = simpledialog.askstring("Unknown Person Name", "Enter the name of the unknown person:")
#                         if unknown_person_name:
#                             unknown_person_folder = os.path.join(dataset_path, unknown_person_name)
#                             os.makedirs(unknown_person_folder, exist_ok=True)
#                             capture_images = True

#                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#                     y = top + 15
#                     cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

#                     cv2.imshow("Capture Unknown Person", frame)

#                     if cv2.waitKey(1) == 27:
#                         cv2.destroyWindow("Capture Unknown Person")
#                         capture_images = False
#                         start_time = time.time()
#                         unrecognized_timer = 0
#                 else:
#                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#                     y = top + 15
#                     cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#             else:
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#                 y = top + 15
#                 cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

#             # Print the recognized person's name in the terminal
#             print("Recognized person:", name)

#         cv2.imshow("Frame", frame)

#         if cv2.waitKey(1) == 27:
#             break

#         if capture_images:
#             image_counter = 0
#             trainer = FaceRecognitionTrainer("DATASET", "treinamento.pickle")
#             unrecognized_timer += 1
#             if unrecognized_timer % 10 == 0:
#                 if image_counter < 100:
#                     image_filename = f"{unknown_person_name}_{time.strftime('%Y%m%d%H%M%S')}_{unrecognized_timer}.jpg"
#                     image_path = os.path.join(dataset_path, unknown_person_name, image_filename)
#                     cv2.imwrite(image_path, frame)
#                     image_counter += 1
#                 elif image_counter > 100:
#                     trainer.train()
                    
        

#         if time.time() - start_time >= unrecognized_time:
#             if not capture_images:
#                 root = tk.Tk()
#                 root.withdraw()
#                 name_input = simpledialog.askstring("Person's Name", "Not recognized for 30 seconds. Enter the person's name:")
                
#                 if name_input:
#                     unknown_person_name = name_input
#                     unknown_person_folder = os.path.join(dataset_path, unknown_person_name)
#                     os.makedirs(unknown_person_folder, exist_ok=True)
#                     capture_images = True

#                     image_filename = f"{unknown_person_name}_{time.strftime('%Y%m%d%H%M%S')}.jpg"
#                     image_path = os.path.join(dataset_path, unknown_person_name, image_filename)
#                     cv2.imwrite(image_path, frame)

#                     train_dataset(dataset_path, encoding_file)

#                     start_time = time.time()

#     cv2.destroyAllWindows()
#     cap.release()


# if __name__ == "__main__":
#     main()


import imutils
import numpy as np
import pickle
import cv2
import face_recognition
import os
import time
import tkinter as tk
from tkinter import simpledialog

def train_dataset(dataset_path, encoding_file):
    image_paths = list(paths.list_images(dataset_path))
    known_encodings = []
    known_names = []

    for image_path in image_paths:
        name = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    data = {"Treinamento": known_encodings, "nomes": known_names}
    with open(encoding_file, "wb") as f:
        pickle.dump(data, f)

    print("Treinamento concluído.")

def main():
    encoding_file = "treinamento.pickle"
    dataset_path = "DATASET"
    unrecognized_time = 30  # Tempo em segundos

    if not os.path.isfile(encoding_file):
        train_dataset(dataset_path, encoding_file)

    data = pickle.loads(open(encoding_file, "rb").read())
    print(data)

    cap = cv2.VideoCapture(1)
    start_time = time.time()
    unrecognized_timer = 0
    capture_images = False
    unknown_person_name = ""

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

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

        for encoding in encodings:
            matches = face_recognition.compare_faces(np.array(encoding), np.array(data["Treinamento"]))
            name = "Desconhecido"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["nomes"][i]
                    counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            if name == "Desconhecido":
                if time.time() - start_time >= unrecognized_time:
                    if not capture_images:
                        root = tk.Tk()
                        root.withdraw()
                        unknown_person_name = simpledialog.askstring("Nome da Pessoa Desconhecida", "Insira o nome da pessoa desconhecida:")
                        if unknown_person_name:
                            unknown_person_folder = os.path.join(dataset_path, unknown_person_name)
                            os.makedirs(unknown_person_folder, exist_ok=True)
                            capture_images = True

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    y = top + 15
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    cv2.imshow("Capture Unknown Person", frame)

                    if cv2.waitKey(1) == 27:
                        cv2.destroyWindow("Capture Unknown Person")
                        capture_images = False
                        start_time = time.time()
                        unrecognized_timer = 0
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    y = top + 15
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Imprime o nome da pessoa reconhecida no terminal
            print("Pessoa reconhecida:", name)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

        if capture_images:
            unrecognized_timer += 1
            if unrecognized_timer % 10 == 0:
                image_filename = f"Unknown_{time.strftime('%Y%m%d%H%M%S')}_{unrecognized_timer}.jpg"
                image_path = os.path.join(dataset_path, unknown_person_name, image_filename)
                cv2.imwrite(image_path, frame)

        if time.time() - start_time >= unrecognized_time:
            if not capture_images:
                root = tk.Tk()
                root.withdraw()
                name_input = simpledialog.askstring("Nome da Pessoa", "Não reconhecido por 30 segundos. Insira o nome da pessoa:")
                
                if name_input:
                    unknown_person_name = name_input
                    unknown_person_folder = os.path.join(dataset_path, unknown_person_name)
                    os.makedirs(unknown_person_folder, exist_ok=True)
                    capture_images = True

                    image_filename = f"{unknown_person_name}_{time.strftime('%Y%m%d%H%M%S')}.jpg"
                    image_path = os.path.join(dataset_path, unknown_person_name, image_filename)
                    cv2.imwrite(image_path, frame)

                    train_dataset(dataset_path, encoding_file)

                    start_time = time.time()

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
