import numpy as np
import cv2
import pickle

# faces.py projenin öğrenme verilerinden yola çıkarak tanıma işleminin gerçekleştirildiği kısımdır. Burada kamera çalıştırılır ve çalışma esnasında tanıma grafikleri kullanılır. Proje içerisinde bulunan "cascades" klasörü OpenCV directory'sinden alınmış ve yüz tanıma değerlendirme metotlarını içerir. 7. Satırda face_cascade'i olarak frontalface(yüzü önden tanıma işlemleri) seçilmiştir ve bu metoda göre yüz tanıma uygulanacaktır.

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')

# face_train.py içerisinde training işlemlerinin ardından kaydedilen verinin burada kullanılması:
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# face_train.py içerisinde verileri pickle ile dizine kaydetmiştik. Burada dizin içerisindeki pickle kullanılarak verilere yeniden erişiyoruz.
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(-1)  # linux -1 de çalışıyor

# Kare kare olacak şekilde kameradan sürekli veri çekilmesi sağlanır...
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # scaleFactor ve minNeigbors OpenCV projelerinde önerilen veriler olan 1.5 ve 5 olarak kullanılmıştır.
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    # Burada kameradan gelen yüz verisi sayısal değerlere çevirilir. Yüz verisi olarak sadece "Region of Interest" gelmektedir. Region of Interest ayrımının yapılmasında OpenCV yardımı alınmaktadır. x, y, w, h(x-koordinat, y-koordinat, width, height) olarak kullanıcının kamerasından gelen görüntü verisi sayısal veriye dönüştürülür.
    for(x, y, w, h) in faces:
        # Burada Region of Interest grayscale formatta alınır. "y:y+h, x:x+w" kırpma işlemini temsil eder. Tüm resimden Region of interest kırpılır.
        roi_gray = gray[y:y+h, x:x+w]
        # Burada Region of Interest renkli formatta alınır. "y:y+h, x:x+w" kırpma işlemini temsil eder. Tüm resimden Region of interest kırpılır.
        roi_color = frame[y:y+h, x:x+w]

        # Grayscale formattaki resimler "face_train.py" içerisinde Numpy array'e çevirilmişti. Şimdi o veriler ve LBPH yüz tanıma algoritması kullanılarak Prediction işlemleri burada yapılacaktır.
        id_, conf = recognizer.predict(roi_gray)
        # "conf" değişkeni algoritmanın benzerlik skorudur. Bu değer süzgeci denemeler sonucu optimum olarak 52 bulunmuştur ve 52 olarak uygulanmıştır.
        if conf >= 52:
            # Resim prediction sonucu uygun ise tanıma ekranında verilecek font ve isim işlemleri burada yapılmıştır. "name" kişinin ismini, "color" yazı rengini, "stroke" kalınlığı temsil eder.
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)
        # Dosya dizini içerisinde yazılı olan "my_image.jpg" kamera açık olduğu sürece burada yeniden kaydedilir. Bu resim kıyaslama verisidir. Kayıt esnasında sadece "Region of Interest" alınır.
        img_item = "my_image.jpg"
        cv2.imwrite(img_item, roi_gray)

        # kamerada tanımlama yapan karenin rengi
        color = (255, 0, 0)
        # kamerada tanımlama yapan karenin kalınlığı
        stroke = 2
        # kamerada tanımlama yapan karenin genişliği
        end_cord_x = x+w
        # kamerada tanımlama yapan karenin yüksekliği
        end_cord_y = y+h
        # kamerada tanımlama yapan karenin çizimi
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    # kamerada tanımlama yapan karenin gösterimi
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# taramanın devam etmesi için fonksiyonlar.
cap.release()
cv2.destroyAllWindows()
