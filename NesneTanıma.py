# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 02:08:43 2020

@author: Zeynep
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    frame_weight = frame.shape[1]       #videodaki resim karelerinin genişliğini alır.
    frame_height = frame.shape[0]       #videodaki resim karelerinin boyunu alır.

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False) #resmin boyutları, resmi RGBye dönüştür, resmi kırpmaya gerek yok

    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
              "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
              "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
              "sofa", "pottedplant", "bed", "diningtable", "toillet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]


    colors = ["0,255,255", "0,0,255", "255,0,0", "255,255,0", "0,255,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18,1))


    model = cv2.dnn.readNetFromDarknet("C:/Users/Zeynep/Desktop/YOLO/pretrained_model/yolov3.cfg","C:/Users/Zeynep/Desktop/YOLO/pretrained_model/yolov3.weights")
    
    layers = model.getLayerNames() #layers ile bütün katmanları çektik.
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
    
    detection_layers = model.forward(output_layer)

    ########## Non-Maximum Suppression - OPERATION 1 ##########
    
    ids_list = [] # id'lerin tutulduğu boş liste
    boxes_list = [] # boxes'ların tutulduğu boş liste
    confidences_list = [] # confidence'lerin tutulduğu boş liste
    
    ########## END OF OPERATION 1 ##########

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            #detection değerlerinden en yüksek skorlu nesnemizi buluruz ->
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence >0.35:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_weight, frame_height, frame_weight, frame_height])
                (box_center_x, box_center_y, box_widht, box_height) = bounding_box.astype("int")
                
                start_x = int(box_center_x - (box_widht/2))
                start_y = int(box_center_y - (box_height/2))
    
                ########## Non-Maximum Suppression - OPERATION 2 ##########
                # yukarıdaki for döngüsünde tespit edilen her şey burada oluşturulan boş listelerin içerine yollandı
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y,int(box_widht), int(box_height)])
    
                ################## END OF OPERATION 2 #####################
                
    ########## Non-Maximum Suppression - OPERATION 3 ##########
    
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4) #en yüksek güvenirliğe sahip dikdörtgenleri gönderir.            
    
    for max_id in max_ids:
        max_class_id = max_id[0]
        box = boxes_list[max_class_id]
        
        start_x = box[0]
        start_y = box[1]
        box_widht = box[2]
        box_height = box[3]
    
        # boundig box'ların üzerine nesne ile ilgili label'ı yazabilmek için;
        predicted_id = ids_list[max_class_id] #max_class_id ile predectid_id ye eriştik
        label = labels[predicted_id] #predicted_id ile label'larımıza eriştik
        confidence = confidences_list[max_class_id] 
    
    ################## END OF OPERATION 3 #####################
    
        end_x = start_x + box_widht
        end_y = start_y + box_height
    
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
                
        label = "{}: {:.2f}½".format(label, confidence*100)
        print("predicted object {}".format(label))
    
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)            
        cv2.putText(frame, label, (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                
    cv2.imshow("Detection Window", frame)
        
    if cv2.waitKey(1) &0xFF == ord("q"):
        break

cap.relase()
cv2.destroyAllWindows()            
            
            
    




