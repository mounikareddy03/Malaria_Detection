import keras
import numpy as np
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras
failed_list=[]
def main(data_path,model_path):
    model = load_model(model_path)
    image=0
    total_imgs=len(os.listdir(data_path))
    print("-"*100)
    for i in (os.listdir(data_path)):
        path=os.path.join(data_path,i)
        img=cv2.imread(path)
        img=cv2.resize(img,(128,128))
        img=img/255
        img = np.reshape(img,[-1,128,128,3])
        
        image += 1

        print("-"*100)
        print("Predicting image : ", image, "/", total_imgs, ":")
        classes = model.predict_classes(img)
        print(classes)
        if classes == 0 :            
            print("Parasitied",classes)
        elif classes == 1:
            print("uninfected",classes)
        else:
            print("Not Predicted")

        if i[0:3]=="par" and classes==0:
            result="correct"
        elif i[0:3]=="uni" and classes==1:
            result="correct"
        else:
            result="wrong"
            failed_list.append(i)
        
        print("Result",result)
    if len(failed_list)!=0:
        print("Failed To Images List: ",failed_list,"Total Number of Failed Images: ",len(failed_list))
    
if __name__ == "__main__":
    data_path=sys.argv[1]
    model_path= sys.argv[2]
    main(data_path,model_path)