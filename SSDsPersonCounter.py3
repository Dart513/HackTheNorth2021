import numpy as np
import time
import cv2
import tensorflow as tf

import winsound

import os

from dataclasses import dataclass

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]


numPeople = 0

increased_x = 3


@dataclass
class PersonStruct:
	id: int
	updated: False
	numFailed: int
	boundingBox: tuple

people = []

#count up object for ids
currentId = 0

videoURL = input("Enter camera IP: ")

MODEL_NAME = 'mobilenet.tflite' 
CWD_PATH = os.getcwd()
MODEL_PATH = os.path.join(CWD_PATH, 'mobilenet', MODEL_NAME)

MIN_CONF_THRESHOLD = 0.6


with open(os.path.join(CWD_PATH, 'mobilenet','labelmap.txt')) as f:
	labels = f.read().splitlines()


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads = 10)
interpreter.allocate_tensors()

labels = []

def bb_intersection_over_union(boxA, boxB):
	widthA = boxA[0] - boxA[3]
	widthB = boxB[0] - boxB[3]

	tempboxA = list(boxA)
	tempboxB = list(boxB)

	tempboxA[0] = (boxA)[0] - int((widthA*increased_x)/2)
	tempboxA[3] = (boxA)[3] + int((widthA*increased_x)/2)

	tempboxB[0] = (boxB)[0] - int((widthB*increased_x)/2)
	tempboxB[3] = (boxB)[3] + int((widthB*increased_x)/2)

	boxA = tuple(tempboxA)
	boxB = tuple(tempboxB)

	# determine the coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def bb_intersection_over_smallest(box0, box1):
	# determine the coordinates of the intersection rectangle

	width0 = box0[0] - box0[3]
	width1 = box1[0] - box1[3]

	tempbox0 = list(box0)
	tempbox1 = list(box1)

	tempbox0[0] = (box0)[0] - int((width0*increased_x)/2)
	tempbox0[3] = (box0)[3] + int((width0*increased_x)/2)

	tempbox1[0] = (box1)[0] - int((width1*increased_x)/2)
	tempbox1[3] = (box1)[3] + int((width1*increased_x)/2)

	box0 = tuple(tempbox0)
	box1 = tuple(tempbox1)

	xA = max(box0[0], box1[0])
	yA = max(box0[1], box1[1])
	xB = min(box0[2], box1[2])
	yB = min(box0[3], box1[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	box0Area = (box0[2] - box0[0] + 1) * (box0[3] - box0[1] + 1)
	box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)

	smallestArea = min(box0Area, box1Area)

	return interArea/smallestArea

def isRedundant(bb0, bb1):
	if bb_intersection_over_union(bb0, bb1) > 0.4:
		return True
	return False
	


def updatePersonCounter(boundingBox, width):
	#find center of boundingBox
	global numPeople
	if ((boundingBox[1] + boundingBox[3])/2) < 0:
		numPeople = numPeople - 1
	else:
		numPeople = numPeople + 1

def updatePositions(img, input_data, width):

	global hog, interpreter
	image_resized = cv2.resize(img, (int(width/2), int(height/2)))
	#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	# detect people in the image
	# returns the bounding boxes for the detected objects
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()

	# Retrieve detection results
	tboxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
	classes = interpreter.get_tensor(output_details[1]['index'])[0]
	scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
	boxes = []

	for index, weight in enumerate(scores):
		if weight > MIN_CONF_THRESHOLD and classes[index] == 0: 
			boxes.append(tboxes[index])
		
		
	for box in boxes:  
		cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255, 255, 0), 2)

	#check for overlaps between existing bounding boxes and current ones
	newBoxes = []
	global people
	#first check if boxes overlap original boxes
	
	for person in people:
		personBox = person.boundingBox
		person.updated = False
		for hogBox in boxes:

			if bb_intersection_over_smallest(person.boundingBox, hogBox) > 0.15:
				print("LOL GET REKT")
				newBoxes.append((person.id, hogBox))
				break
		

	for newBox in newBoxes:
		tempPerson = list(filter(lambda p: p.id == newBox[0], people))[0]
	
		if tempPerson:
			print("We found 'em!")
			person.updated = True
			break
			
	for newBox in newBoxes:
		person = list(filter(lambda p: p.id == newBox[0], people))[0]
		person.boundingBox = newBox[1]

	for person in people:
		if not (person.updated):
			person.numFailed = person.numFailed + 1
			
			if (person.numFailed >= 4):
				print("REMOVING LAD")
				updatePersonCounter(getattr(person, 'boundingBox'), width)
				people.remove(person)


	



# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

cv2.startWindowThread()
cap = cv2.VideoCapture(videoURL)

loopNum = 0
lastTime = time.time()

while True:
	print("Num People " + str(numPeople) + ' ' + str(len(people)))

	loopNum = loopNum + 1

	

	ret, img = cap.read()
	#pre-processing image
	#if ((time.time() - lastTime) > 1):
	#	while ret:
	#		ret, img = cap.read()
	#		img1 = img
	#	img = img1

	if not (ret): 
		time.sleep(0.1)
		continue
	
	

	lastTime = time.time()

	#image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	imH, imW, _ = img.shape 
	image_resized = cv2.resize(img, (width, height))
	input_data = np.expand_dims(image_resized, axis=0)

	# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
	if floating_model:
		input_data = (np.float32(input_data) - input_mean) / input_std


	if loopNum % 2 == 0:
	

		# Perform the actual detection by running the model with the image as input
		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		

		# Retrieve detection results
		boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
		classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
		scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

		#detect redundancy between current boxes and tracked boxes
		for index, box in enumerate(boxes):
			#print("trying add")
			if ((scores[index] > MIN_CONF_THRESHOLD) and (scores[index] <= 1.0) and classes[index] == 0):
				#print("passed threshold")
				box = tuple(box)
				for person in people:
					doesRedundant = isRedundant(getattr(person, 'boundingBox'), box)
					print(doesRedundant)
					if not (doesRedundant):
						print("is not redundant")
						#make a new tracker, make a new person
						people.append(PersonStruct(currentId, False, 0, box))
						currentId = currentId + 1

				if len(people) == 0:
					#print("Really adding") 

					people.append(PersonStruct(currentId, False, 0, box))
					currentId = currentId + 1

	else:
		updatePositions(image_resized, input_data, width)
			
		

	for index, person in enumerate(people):

		box = getattr(person, "boundingBox")

		# Get bounding box coordinates and draw box
		# Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
		ymin = int(max(1,(box[0] * imH)))
		xmin = int(max(1,(box[1] * imW)))
		ymax = int(min(imH,(box[2] * imH)))
		xmax = int(min(imW,(box[3] * imW)))
		
		cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

		# Draw label
		#object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
		#label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
		#labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
		#label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
		#cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
		#cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


		#print(classes)


	if numPeople > 0:
		winsound.Beep(2500, 200)
	cv2.imshow('frame', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
