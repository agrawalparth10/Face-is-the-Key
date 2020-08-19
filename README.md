# Face-is-the-Key

### Description
Encrypt and Decrypt Files using Face Recognition. 

### TODO List

1. -[x] ~~Use Caffe Model for Face Detection~~
2. -[x] ~~Use OpenFace pretrained models for Face Recognition~~
3. -[x] ~~Create a class to encrypt and decrypt files~~
4. -[x] ~~Train SVM for classification~~ 
5. -[x] ~~Combine all the different classes~~
6. -[ ] Develop training scripts for Face Detection
7. -[ ] Develop training scripts to create embeddings - ArcFace, VarGFaceNet 
8. -[ ] Safely store the Key

### Usage
**Create Database**

Store a few images of the user in ./dataset/user folder
```
cd dataset
cd user
```

Store a few images of random people in ./dataset/unknown folder
```
cd ..
cd unknown
```

**Create Embeddings**
To create embeddings for the user, run the following command - 
```
python face_detection.py
```
This will store the embeddings in the pickle folder

**Train Classification Model** 
To train the SVM classification model, run the following command - 
```
python model.py
```

**Encrypt and Decrypt Files**
To encrypt files - 
```
python main.py -f [FILE PATH] -m encrypt
```

To decrypt files - 
```
python main.py -f [FILE NAME] -m decrypt
```
The face of the user will be recognized using the webcam. If the user is matched, the file will be decrypted. 

### References 
1. <https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/>
2. <https://arxiv.org/abs/1503.03832>
3. <https://github.com/deepinsight/insightface>
4. <https://github.com/cmusatyalab/openface>
5. <https://www.youtube.com/watch?v=UB2VX4vNUa0>
6. <https://arxiv.org/abs/1910.04985>
