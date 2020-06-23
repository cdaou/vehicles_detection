import cv2, time
import numpy as np

video_capture = cv2.VideoCapture("/dev/video0")
video_capture.set(3, 480)
video_capture.set(4, 320)
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setShadowValue(False)
fgbg.setVarThreshold(8)

min_car_pixels = 110
max_car_pixels = 500

if video_capture.isOpened():
	ret, frame = video_capture.read()
	start_time = time.time()
	objects = 0;
	while ret:
		fgbg.setBackgroundRatio(0.1)
		ret, frame = video_capture.read()
		blur_img = cv2.GaussianBlur(frame, (3,3),0)
		gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
		fgmask = fgbg.apply(gray)
		cv2.imshow('fgmask', fgmask)	
		
		# DRAW RECTANGLES ON DETECTED CARS
		contours, hier = cv2.findContours(fgmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
		for contour in contours:
				area = cv2.contourArea(contour)
				if area > min_car_pixels and area < max_car_pixels :
					(x,y,w,h) = cv2.boundingRect(contour)
					cv2.rectangle(frame, (x-5,y-5), (x+w+5,y+h+5), (0, 255, 0), 2)
		# GREEN LIGHT
		if(time.time()-start_time < 5):
			cv2.putText(frame, "The Traffic Light is GREEN", (10,20),cv2.LINE_AA, 0.6, (0,255,0), 1)
			cv2.putText(frame, "In the previous red light there were {} vehicles".format(objects), (10,40), cv2.LINE_AA, 0.4, (0,0,0), 1)
			cv2.imshow('Original Video', frame)	
		# RED LIGHT	
		elif(time.time()-start_time > 5 and time.time()-start_time < 15):
			fgbg.setBackgroundRatio(0.0001)
			cv2.putText(frame, "The Traffic Light is RED", (10,20),cv2.LINE_AA, 0.6, (0,0,255), 1)
			cv2.imshow('Original Video', frame)
			# RIGHT BEFORE RED LIGHT ENDS
			if(time.time()-start_time < 14):
				cv2.imwrite('/home/pi/Desktop/cars.jpeg', fgmask)
				# SAVE NUMBER OF VEHICLES
				objects = 0
				for contour in contours:
					area = cv2.contourArea(contour)
					if area > min_car_pixels and area < max_car_pixels :
						objects=objects+1
				cv2.imshow('Original Video', frame)	
		# INITILIZATION AFTER GREEN+RED		
		else:
			start_time = time.time()

		k = cv2.waitKey(20)   
		if k%256 == 27 & 0xFF: 
			cv2.destroyWindow("Original Video")
			cv2.destroyWindow("fgmask")
			break
else:
	print("There is a problem with camera capture")
	
video_capture.release()
cv2.destroyAllWindows()
