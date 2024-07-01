import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import cv2
import random



def show_mask(mask, image,random_color=True ):
    if random_color:
        color = random.sample(range(1, 255), 3)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    #color = color.reshape(1, 1, 3)
    h, w = mask.shape[-2:]
    mask.dtype = np.uint8()
    mask = mask.reshape(h, w, 1)
    
    image[(mask==1).all(-1)] = color
    return image
    

    
  


   
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box , name , image):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    image = cv2.rectangle(image , (int(x0)+5, int(y0)+5), (int(x0+w), int(y0+h)), [0,0,255] ,  2)  
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image ,name, (int(x0)+5,int(y0)+25) , font , 1 , [0,0,0] , 2 ,  cv2.LINE_AA)
    return image

model = YOLO('SAM_YOLO8//yolov8n.pt')
# run the model on the image
results = model.predict(source='Segment_anything//dog.jpg', conf=0.25)[0]



bbox = results.boxes.xyxy.tolist()
names = []
for i in results.boxes.cls.cpu().detach().numpy().tolist():
  names.append(results.names[i])




image = cv2.cvtColor(cv2.imread('Segment_anything//dog.jpg'), cv2.COLOR_BGR2RGB)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_b"](checkpoint="Segment_anything//sam_vit_b_01ec64.pth").to(device=DEVICE)
mask_predictor = SamPredictor(sam)
mask_predictor.set_image(image)



counter = 0 
for boz in bbox:
  input_box = np.array(boz)
  masks, _, _ = mask_predictor.predict(
      point_coords=None,
      point_labels=None,
      box=input_box[None,:],
      multimask_output=False,
  )

  
  image = show_mask(masks[0] , image )
  image = show_box(input_box  , names[counter] , image)
  counter+=1


cv2.imshow("Output  : " , image)
cv2.waitKey(0)





