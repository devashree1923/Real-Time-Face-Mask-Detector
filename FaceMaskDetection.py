import cv2 
import uuid
import json
from ibm_watson import VisualRecognitionV4
from ibm_watson.visual_recognition_v4 import FileWithMetadata, AnalyzeEnums
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from matplotlib import pyplot as plt

#  Image capture
cap = cv2.VideoCapture(2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True: 
    ret, frame = cap.read()
    imgname = './Images/No Mask/{}.jpg'.format(str(uuid.uuid1()))
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()

apikey = 'YOUR API KEY HERE'
url = 'YOUR URL HERE'
collection = 'YOUR COLLECTION HERE'

authenticator = IAMAuthenticator(apikey)
service = VisualRecognitionV4('2018-03-19', authenticator=authenticator)
service.set_service_url(url)

path = 'PATH TO YOUR IMAGE'

with open(path, 'rb') as mask_img:
    analyze_images = service.analyze(collection_ids=[collection], 
                                     features=[AnalyzeEnums.Features.OBJECTS.value], 
                                    images_file=[FileWithMetadata(mask_img)]).get_result()



obj = analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['object']
coords = analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['location']

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, text=obj, org=(coords['left']+coords['width'], coords['top']+coords['height']), fontFace=font, fontScale=2, color=(0,255,0), thickness=5, lineType=cv2.LINE_AA)
plt.imshow(img)
