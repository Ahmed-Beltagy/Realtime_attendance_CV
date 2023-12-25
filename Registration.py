
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import os
import cv2
from PIL import Image
import csv
import time
from datetime import datetime
import pickle

st.set_page_config(page_title="Realtime FR Registration", page_icon=":fire:", layout="wide", initial_sidebar_state="expanded")
path = st.text_input("Enter the path where the data files will be saved:")

def GetImgID(path):
    imagesPath = [os.path.join(path + "//Dataset", i) for i in os.listdir(path+"//Dataset")]
    Faces = []
    IDs = []
    
    for img in imagesPath:
        face = Image.open(img).convert("L")
        faceNumpy = np.array(face, "uint8")
        
        #Extract Id
        Id = os.path.split(img)[-1].split(".")[1]
        Id = int(Id)
        Faces.append(faceNumpy)
        IDs.append(Id)
        
        cv2.imshow("Training Progress....", faceNumpy)
        cv2.waitKey(1)
    return IDs, Faces



def main():
    Users = {}
    with st.sidebar:
        selected = option_menu(
        menu_title="Dashboard",
            options=["Registeration", "Training", "Testing"]
        )
    
    if selected == "Registeration":
        st.title("Student Registration for Real Time Face Recognition Attendance")
        st.write("Please enter your name and ID below:")
        name = st.text_input("Name")
        ID = st.text_input("ID")
        if name and ID and path:
            st.write("Path:", path)
            st.write("Name:", name)
            st.write("ID:", ID)
            st.write("NOTE: The camera will now open to take images of your face with different positions.")
            if st.button("Accept"):
                Users[ID] = name
                with open(path + "//Users.pkl", "wb") as f:
                    pickle.dump(Users, f)
                cap = cv2.VideoCapture(0)
                trained_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                count = 0
                while True:
                    _, frame = cap.read()
                    gray_scale2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = trained_data.detectMultiScale(gray_scale2, 1.3, 5)
                    for x,y,w,h in faces:
                        count = count + 1
                        cv2.imwrite(path + "//Dataset//User." + str(ID)+"."+str(count)+".jpg", gray_scale2[y:y+h, x:x+w])
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
                    cv2.imshow("Camera", frame)
                    k = cv2.waitKey(1)
                    if count > 600:
                        break
                cap.release()
                cv2.destroyAllWindows()
                st.write("Completed, Thank You.")
                
    elif selected == "Training":
        st.empty()
        st.empty()
        st.empty()
        st.empty()
        st.title("Training in progress........")
        st.write("Loading........")
        trainer = cv2.face.LBPHFaceRecognizer_create()
        Ids, faces = GetImgID(path)
        trainer.train(faces, np.array(Ids))
        trainer.write(path + "//Trainer.yml")
        cv2.destroyAllWindows()
        st.write("Completed.......")
        st.title("Training Completed successfuly!")

    else:
        st.empty()
        st.empty()
        st.empty()
        st.empty()
        with open(path + "//Users.pkl", "rb") as f:
            Users = pickle.load(f)

        Column_Names = ["Name", "ID", "Entered_Date", "Entered_Time"]

        st.title("Camera is Opening...")
        
        cap = cv2.VideoCapture(0)

        trained_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        trainer = cv2.face.LBPHFaceRecognizer_create()

        #Import Trained Model

        trainer.read(path + "//Trainer.yml")


        # Dictionary to store the last attendance time for each person
        last_attendance_time = {}

        while True:

            _, frame = cap.read()

            gray_scale2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = trained_data.detectMultiScale(gray_scale2, 1.3, 5)

            for x,y,w,h in faces:

                Id, conf = trainer.predict(gray_scale2[y:y+h, x:x+w])

                print(conf)

                if conf < 50:

                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)

                    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)

                    cv2.putText(frame, Users[str(Id)], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

                    tnow = time.time()

                    date = datetime.fromtimestamp(tnow).strftime("%d-%m-%Y")

                    timest = datetime.fromtimestamp(tnow).strftime("%H:%M:%S")

                    # Check if attendance was already taken for this person in the last 5 minutes
                    if Id in last_attendance_time and tnow - last_attendance_time[Id] < 60:
                        continue

                    exist = os.path.isfile(path + "//Attendance//Attendance_"+date+".csv")

                    attendance = [str(Users[str(Id)]), str(Id), str(date), str(timest)]

                    # Update the last attendance time for this person
                    last_attendance_time[Id] = tnow

                    if exist:
                        with open(path + "//Attendance//Attendance_"+date+".csv", "+a") as file:
                            adder = csv.writer(file)
                            adder.writerow(attendance)
                        file.close()
                    else:
                        with open(path + "//Attendance//Attendance_"+date+".csv", "+a") as file:
                            adder = csv.writer(file)
                            adder.writerow(Column_Names)
                            adder.writerow(attendance)
                        file.close()

                else:

                    cv2.putText(frame, "Not Registered", (x, y-40), cv2.FONT_ITALIC, 1, (50,50,255), 2)

                    cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) == ord('q'):

                break

        cap.release()

        cv2.destroyAllWindows()

        st.title("Test Completed!")
        

main()
