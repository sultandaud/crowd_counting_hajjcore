import streamlit as st
from PIL import Image
import torch
from models import vgg19
import gdown
from torchvision import transforms
import cv2
import numpy as np
import scipy
from scipy.io import loadmat
import csv
import matplotlib.pyplot as plt

st.title("Crowd Counting App")
st.markdown('This work is supported by Hajj and Umrah Research Cente, Umm Al-Qura University') 
# Define the points of the bounding boxes
box1_points = [(466, 176), (600, 210)]
box2_points = [(542, 250), (685, 285)]

count_bbox=[(421,12),(235,66)]


model_path = "pretrained_models/model_qnrf.pth"
device = torch.device(0)  # device can be "cpu" or "gpu"
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()

# Radio button to select the display mode
display_mode = st.radio("Select Display Mode:", ["Main-Dashboard","Location-1","Location-2"])
frame_number=[]
loc_count1=[]
loc_count2=[]


map_placeholder = st.empty()
graph_placeholder = st.empty()
image1_placeholder = st.empty()
image2_placeholder = st.empty()

denmap1_placeholder = st.empty()
denmap2_placeholder = st.empty()
 
# Function to draw bounding boxes and display values
def draw_bounding_boxes(image, value1, value2):
    # Draw the bounding boxes on the image
    cv2.rectangle(image, box1_points[0], box1_points[1], (0, 255, 0), 2)
    cv2.rectangle(image, box2_points[0], box2_points[1], (0, 255, 0), 2)

    # Display the random values inside the bounding boxes
    cv2.putText(image, value1, (box1_points[0][0] + 10, box1_points[0][1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(image, value2, (box2_points[0][0] + 10, box2_points[0][1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return image


def predict(inp):
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    inp = inp.to(device)
    with torch.set_grad_enabled(False):
        outputs, _ = model(inp)
    count = torch.sum(outputs).item()
    vis_img = outputs[0, 0].cpu().numpy()
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img, int(count)

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def generate_data(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    # mat_path = im_path.replace('.jpg', '_ann.mat')
    # points = loadmat(mat_path)['annPoints'].astype(np.float32)
    # idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    # points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        # points = points * rr
    return Image.fromarray(im)



def main():
       
    image = cv2.imread('map.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cap1 = cv2.VideoCapture('jamarat2.mp4')
    cap2 = cv2.VideoCapture('jamarat3.mp4')
    image_placeholder1 = st.empty()
    image_placeholder2 = st.empty()
    for i in range(100):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        pil_frame1 = Image.fromarray(frame1)
        pil_frame2 = Image.fromarray(frame2)
        vis_img1,count1 = process_image(pil_frame1)
        vis_img2,count2 = process_image(pil_frame2)
        if display_mode == "Main-Dashboard":
            bounding_box_display(count1,count2,image)
            
        if display_mode == "Location-1":
            display_location1(frame1,vis_img1,count1)
            
        if display_mode == "Location-2":
            display_location1(frame2,vis_img2,count2)
        
def display_location1(frame1,vis_img1,count1):
    
    #cv2.rectangle(frame1, count_bbox[0], count_bbox[1], (0, 255, 0), 2)
    cv2.putText(frame1, "Loc1-Count:"+str(count1), (count_bbox[0][0] + 10, count_bbox[0][1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 4)
    image1_placeholder.image(frame1, use_column_width=True)
    st.empty()
    denmap1_placeholder.image(vis_img1,use_column_width=True)
    st.empty()
    


def display_location2(frame2,vis_img2,count2):
    #cv2.rectangle(frame2, count_bbox[0], count_bbox[1], (0, 255, 0), 2)
    cv2.putText(frame2, "Loc2-Count:"+str(count2), (count_bbox[0][0] + 10, count_bbox[0][1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    image2_placeholder.image(frame2, use_column_width=True)
    st.empty()
    denmap2_placeholder.image(vis_img2,use_column_width=True)
    st.empty()
        
def bounding_box_display(count1,count2,image):
    
    #map_placeholder = st.empty()
    #graph_placeholder = st.empty()
    fig, ax = plt.subplots()
    frame_number.append(len(frame_number) + 1)
    loc_count1.append(count1)
    loc_count2.append(count2)
    ax.clear()
    # Plot the first curve
    ax.plot(frame_number, loc_count1, label='Location-1')
    ax.plot(frame_number, loc_count2, label='Location-2')
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Count")
    ax.set_title("Live Graph")
    ax.legend()
    graph_placeholder.pyplot(fig)
    st.empty()
    image_with_boxes = draw_bounding_boxes(np.copy(image), "Loc1:"+str(count1), "Loc2:"+str(count2))
    map_placeholder.image(image_with_boxes, caption=f'Image with Bounding Boxes and Values', use_column_width=True)
    st.empty()  

  

	
def process_image(image):
    min_size = 512
    max_size = 2048
    im_w, im_h = image.size
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(image)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
    np_im = np.array(im)
    vis_img, count =  predict(np_im)
    return vis_img, count

if __name__ == "__main__":
    main()
