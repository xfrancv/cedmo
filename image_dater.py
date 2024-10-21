# 
import os
from PIL import Image
import datetime
import torch
import numpy as np
from facis.helpers.bbox.utils import draw_labeled_bbox, crop_bbox
from facis.faces.deployment.app import FacePredictor
from facis.helpers.utils import MyYamlLoader
import cv2
import insightface
import matplotlib.pyplot as plt

def get_face_description(predictions, attributes, attrib_name=False):
    '''
    Convert a dictionary of predcited attributes to string.
    '''
        
    annotations = []
    for face in predictions:
        if not attrib_name:
            text = []
            for item in attributes:
                if item not in face:
                    continue
                if len(text) == 0:
                    text = f"{face[item]}"
                else:
                    text = text + f",{face[item]}"
            text = [text]
        else:
            text = []
            for item in attributes:
                text.append(f"{item}: {face[item]}")
        annotations.append(text)
    return annotations


def is_in_bbox(x, y, bbox):
    ''' 
    The function checks if the point (x, y) is inside the bbox which is
    represented as a rectangle given by four points.
    '''
    x = np.array((x,y))    
    points = [np.array((p[0],p[1])) for p in bbox]

    for i in range(len(points)):
        a = points[i]
        b = points[(i+1)%len(points)]
        t = np.dot(x-b, a-b)/np.dot(a-b,a-b)
        if t < 0 or t > 1:
            return False
        
    return True    


def get_identity_templates( app ):
    
    # go over subdirectories in "data/", load all images and represent them as templates
    names = []
    templates = []
    birth_dates = []
    for subdir in os.listdir("data"):

        str = subdir.split("_")
#        names.append(f"{str[0]} {str[1]}")
        names.append(f"{str[0]}")
        birth_dates.append(int(str[2]))
        
        dscr = []
        for file in os.listdir(f"data/{subdir}"):
            face_path = f"data/{subdir}/{file}"
            face_img = cv2.imread(face_path)

            faces = app.get(face_img)
            for face in faces:
                f = face.embedding
                norm = np.linalg.norm(f)
                f = f / norm
                dscr.append(f)

        print(f"name: {names[-1]}, birthdate: {birth_dates[-1]}, #faces: {len(dscr)}")

        # convert a list of numpy vectors to numpy arryy
        dscr = np.array(dscr)                            
        template = np.mean( dscr, axis = 0, keepdims=True )
        norm = np.linalg.norm(template, axis=1, keepdims=True)
        template = template / norm
        templates.append(template)  

        #    rimg = app.draw_on(in_img, faces)
        #        cv2.imwrite(f"./t1_output_{file}", rimg)

    templates = np.array(templates) 
    templates = np.squeeze(templates, axis=None)

    return names, templates, birth_dates

if __name__ == '__main__':

    ##
#    image_path = "images/04005471.jpeg"
#    image_path = "images/zeman_2024.png"
    #image_path = "images/milos_zeman_with_wife.jpeg"
#    image_path = "images/zemanem_necas_2013.jpg"
#    image_path = "images/zeman_putin_2017.jpg"
#    image_path = "images/putin_minsk_2015.png"
    image_path = "images/putin_makron_merkel_2017.png"

    # load image as np array
    in_img = cv2.imread(image_path)

    face_engine = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_engine.prepare(ctx_id=0, det_size=(640, 640))

    # go over directory and load all images
    names, templates, birth_dates = get_identity_templates(face_engine)

    #
    faces = face_engine.get(in_img)
    rimg = face_engine.draw_on(in_img, faces)
    #cv2.imwrite("./output.jpg", rimg)

    ## init facis prediction model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor_model = "models/000_model_cpu.pt"
#    predictor_model = "models/model_distilled_cpu.pt"
    #predictor_model = "models/cpu_age_ensemble.pt"
    model = FacePredictor(predictor_model, device)

    ## find faces and predict attributes
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    model.recognize(in_img)

    # get boxes, attributes and landmarks
    bboxes = model.get_bboxes()
    predictions = model.get_attributes(attribute_subset=["age_pred"])
    #landmarks = model.get_landmarks()

    # match landmarks with bboxes and get face descriptors
    face_descriptors  = []
    for bbox in bboxes:
        center = np.mean(np.array(bbox),axis=0)
        descr = None
        for face in faces:
            bbox2 = [(face.bbox[0], face.bbox[1]), (face.bbox[2], face.bbox[1]), (face.bbox[2], face.bbox[3]), (face.bbox[0], face.bbox[3])]
            if is_in_bbox(center[0], center[1], bbox2):
                descr = face.embedding
                norm = np.linalg.norm(descr)
                descr = descr / norm
                break
        if descr is None:
            print("No descriptor found.")
        face_descriptors.append(descr)        

    # match descriptors with templates using cosine similarity
    similarity_threshold = 0.6
    for i, descr in enumerate(face_descriptors):
        if descr is None:
            continue

        cosine_similarity = np.matmul(descr,np.transpose(templates))
        idx = np.argmax(cosine_similarity)
        if cosine_similarity[idx] > similarity_threshold:
#            predictions[i]["name"] = f"{names[idx]}({cosine_similarity[idx]:.1f})"
            predictions[i]["name"] = f"{names[idx]}"
            predictions[i]["birthdate"] = birth_dates[idx]

    # find the most likely creation year
    idx = []
    for i, prediction in enumerate(predictions):
        if "name" in prediction:
            idx.append(i)
        else:
            predictions[i]["name"] = "???"

    if len(idx) > 0:
        years = np.linspace(1990, 2024, 2024-1990+1, dtype=int)
        score = np.zeros(len(years))
        for i, y in enumerate(years):
            score[i] = 0
            for j in idx:
                score[i] += np.abs(y - predictions[j]["birthdate"]-predictions[j]["age_pred"])
                
    # draw labeled boxes
    out_img = in_img.copy()
    out_img = Image.fromarray(in_img)    
    descriptions = get_face_description(predictions, ["age_pred","name"], attrib_name=False)        
    for i, text in enumerate(descriptions):
        draw_labeled_bbox(out_img, bboxes[i], text)

    # save image
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    out_img_path = os.path.dirname(image_path) + "/" + file_name + "_out.png"
    out_img.save(out_img_path)
    out_img.show()

    # save years vs. score plot
    out_img_path = os.path.dirname(image_path) + "/" + file_name + "_creation_year.png"
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(10, 5)) 
    plt.plot(years, score)
    plt.xlabel("Year")
    plt.grid()
    plt.savefig(out_img_path)






    
