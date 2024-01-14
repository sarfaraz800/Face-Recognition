!pip install numpy opencv-python
!pip install dlib
!pip install face_recognition
import face_recognition as fr
import cv2
import numpy as np
import os
from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/train'
known_names = []
known_name_encodings = []
images = os.listdir(path)
for img in images:
  image = fr.load_image_file('/content/drive/MyDrive/train/'+img)
  image=cv2.resize(image,(864,1152))
  lm=fr.face_landmarks(image)
  #print(lm)
  if not lm:
    print(img,"Face not detected")
    #print(lm)
  else:
    image_path = '/content/drive/MyDrive/train/'+img
    encoding = fr.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))

  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))


test_image = '/content/photo.jpg'
image = cv2.imread(test_image)
face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)
recognized_name=[]
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
   matches = fr.compare_faces(known_name_encodings, face_encoding)
   #print(matches)
   name = ""

   face_distances = fr.face_distance(known_name_encodings, face_encoding)
   best_match = np.argmin(face_distances)

   if matches[best_match]:
       name = known_names[best_match]
       #print(name)
       recognized_name.append(name)
   else:
     print("face not recognized Please! try again")

   cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
   cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)

   font = cv2.FONT_HERSHEY_DUPLEX
   cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
from google.colab.patches import cv2_imshow
cv2_imshow(image)
attend={'NABEEL':'A\n','AMAN':'A\n','SARFARAZ':'A\n','ARUN':'A\n','FARDEEN':'A\n','AFIFA':'A\n','SUHAS':'A\n','RATAN':'A\n'}
attend_list=open("/content/attendance_list.txt","w")
attend_list.write("Name        Status\n")
attend_list.write("__________________\n")
for name in recognized_name:
  name=name.upper()
  if name in attend.keys():
    attend[name]='P\n'
    #attend_list.write(name+'     ')
    #attend_list.write(attend[name]+'\n')
  else:
    print(name,"is not in the list")
    attend[name]='P\n'
    print("Your name is added to list.")
#attend=str(attend)
#attend_list.write(attend)
max=12
for name in attend.keys():
  l=len(name)
  white_space=max-l
  attend_list.write(name)
  for i in range(white_space+1):
    attend_list.write(" ")
  attend_list.write(attend[name])
cv2.imwrite("./output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()