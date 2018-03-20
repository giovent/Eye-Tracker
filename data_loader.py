### data_loader.py ###################
# Contains "load_training_data" that #
# loads the data from data_folder and#
# returns the part of the image with #
# the eyes (unless only_eyes=False)  #
# as well as the labels for each img #
######################################

import numpy as np
import cv2

def load_data(data_folder, show_detailed_process=False, only_eyes=True):
    if(data_folder[len(data_folder)-1]!='/'):
        data_folder+='/'
    total_people = 56
    count = 0
    X_train = []
    y_train = []
    if(show_detailed_process):
        print "Loading data................................................."
    for person_n in range(1,total_people+1,1):
        person_str = "00"+str(person_n)+"_2m_"
        if(person_n<10):
            person_str = "0"+person_str
        for p,P in enumerate([-30,-15,0,15,30]):
            for v,V in enumerate([-10,0,10]):
                for h,H in enumerate([-15,-10,-5,0,5,10,15]):
                    filename=data_folder+person_str+str(P)+"P_"+str(V)+"V_"+str(H)+"H.jpg"
                    img = cv2.imread(filename)
                    if(img is None or img==[]):
                        print "No image found loading data from '..." + filename[len(filename)-10:] + "'"
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img,(100,100))
                        img = img/255.0
                        if(only_eyes):
                            img = img[29:49,16:88]
                        count+=1
                        X_train.append(img)
                        y = np.zeros([2])
                        y[0] = V
                        y[1] = H
                        y_train.append(y)
                        if(show_detailed_process):
                            cv2.imshow("Image processing",img)
                            cv2.waitKey(delay=1)
    if(show_detailed_process):
        print "Successfully loaded " + str(count) +" images from folder " + data_folder
        print "every picture has a shape of ", X_train[0].shape
    return np.vstack(X_train),np.array(y_train)

# This is how the load_data function must be called
#X_train,y_train=load_data("Dataset/Columbia_Processed_100_100", True)
