# Biometrics-face-recognition-lfw
#### OpenCV Biometrics Project

##### Cascade : Open CV frontal face 
##### Dataset : Labeled Faces in the Wild [kaggle link](https://www.kaggle.com/jessicali9530/lfw-dataset)

## Usage
```bash
# create lfw folder inside then put all celebrity named image folders
mkdir lfw

# create train file may take a few minutes 
python3 face_train.py

# then run faces.py
python3 faces.py

```

## Directory File Tree
```bash
.
├── cascades
│   └── data
│       ├── haarcascade_frontalface_alt2.xml
│       ├── ...
│       ├── __init__.py
│       └── __pycache__
│           └── __init__.cpython-37.pyc
├── faces.py
├── face_train.py
├── labels.pickle
├── lfw  
│   ├── AJ_Cook
│   └── ...
├── my_image .jpg
└── trainner.yml
```
## Notes
```py
# open cv video capture properties -1 on linux, 0 on windows
cap = cv2.VideoCapture(-1) 

# project presentation was turkish so comments are in turkish language
```

### Cheers
![Hide The Pain Developer Harold](https://github.com/configtheworld/biometrics-face-recognition-lfw/blob/master/my_image%20.jpg?raw=true)

