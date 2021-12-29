import cv2
import os
import numpy as np
from PIL import Image
import pickle

# face_train.py dosyası projenin machine learning kısmından sorumludur. "labels.pickle", "trainner.yml" dosyaları programın çalışmasıyla oluşturulan dosyalardır. bu dosyalar silinip face_train.py tekrar çalıştırılırsa "lfw" database'ine göre proje yeniden train edilir.


# Dataset directory burada belirlenir:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "lfw")

# OpenCV'den ön yüz cascadeinin(cascadeleri kullanarak ML modelleri yapabiliriz. Directory içerisinde "cascades" klasörü incelendiğinde farklı amaçlar için farklı cascadeler bulunmaktadır.) sağlanması...
face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
# OpenCV'nin sağladığı yüz tanıma: LBPH(Local Binary Pattern Histogram)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Resimler işlenirken kullanılacak veri türleri belirlenir.
# Oluşturulan ve Şu anda işlenen resmin id'si
current_id = 0
# Oluşturulan Id labellarının tutulduğu yer.
label_ids = {}
# Labelların tutulduğu yer
y_labels = []
# Train için işlenen datanın(Numpy array olarak) tutulduğu yer
x_train = []

# Directory for ile gezilir...
for root, dirs, files in os.walk(image_dir):
    for file in files:
        # Directory içerisinde resim uzantı ve isim kontrolleri
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()
            # Resimlerin id'leme işleminin yapılması
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # Burada resimler pixel değerleri(grayscale) ile numpy arraylere çevirilerek matematiksel işlem yapılmasına olanak sağlanır. Bunun öncesinde resimler aynı size'a getirilir ve kenar yumuşatma (anti-aliasing) uygulanır.
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            # faces içerisinde resim "faces.py"ın aksine numpy array baz alınarak tespit edilmektedir.
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.5, minNeighbors=5)

            # "x_train" ve "y_labels"a data sağlayan for döngüsü
            for(x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# Python'ın sağladığı "pickle" ile burada oluşturulan verileri proje path'inde pickle veri türünde tutabiliriz.
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# Daha önce oluşturulan ve dataseti dolaşarak resimleri Numpy array'e çevirerek veriyi tutan değişkenler ile LBPH train işlemi yapılır. Ardından train dosyası directory içerisine kaydedilir.
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
