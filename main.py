import argparse
from encryption import Encryptor
from recognize import Recognizer


ap = argparse.ArgumentParser()
ap.add_argument("-f","--file", required= True, help = "Path to file")
ap.add_argument("-m","--mode", required= True, help = "Enter 'encrypt' or 'decrypt'")
args = vars(ap.parse_args())

file_path = args['file']
mode = args['mode']


key = b'[EX\xc8\xd5\xbfI{\xa2$\x05(\xd5\x18\xbf\xc0\x85)\x10nc\x94\x02)j\xdf\xcb\xc4\x94\x9d(\x9e'
enc = Encryptor(key)
clear = lambda: os.system('cls')

if(mode == "encrypt"):
    try:
        enc.encrypt_file(file_path)
        print("File encrypted")
    except:
        print("Incorrect file path")
        exit()

elif(mode == "decrypt"):
    recog = Recognizer()
    detection = recog.recognize()
    if(detection):
        enc.decrypt_file(file_path + ".enc")
        print("File decrypted")
    else:
        print("Incorrect user")

else:
    print("Incorrect mode")