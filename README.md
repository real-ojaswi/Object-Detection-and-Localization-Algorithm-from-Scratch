## Multi-Object Detection and Localization by Implementing YOLO Logic from Scratch

The repository consists of the code for multi object localization and detection by implementing the YOLO logic from scratch. COCO dataset has been used for training purpose. The script *datageneration.py* consists the code for extraction of images using COCO API. Only three classes have been used, thus forming 9-element yolo tensor for training purpose. The bounding boxes have been trained using MSE Loss and cIOU Loss and then the results are compared. The trainer class that implements the yolo logic has been adapted from the implementation in DLStudio by Prof. Avi Kak (https://engineering.purdue.edu/kak/distDLS/).

Some decent results obtained on test dataset:
![test_bset9](https://github.com/thenoobcoderr/Object-Detection-and-Localization-from-scratch/assets/139956609/0b5444eb-cb08-4e34-9cea-d87c73078096)
![test_best4](https://github.com/thenoobcoderr/Object-Detection-and-Localization-from-scratch/assets/139956609/4e6dc21e-fb56-467c-8106-dc4c70b73a40)

Some not-so-decent results obtained
![test_error3](https://github.com/thenoobcoderr/Object-Detection-and-Localization-from-scratch/assets/139956609/97a3938f-172b-452d-9fff-5a9e42bd9a9e)


The implementation does a great job in single-object detection and localizaiton, but sometimes struggle with localization when there are multiple objects in a frame. It sometimes detects and localizes only one of them or draws the bounding box at an improper location. However, for a simple model trained on a small dataset, and with limited resources, I think it goes over the expectations.
