# COVID Cam TV
This project won Hack The North 2021!!!!


## Inspiration: 
Many businesses are struggling to currently find employees. Many of them don’t want to pay employees to stand outside throughout the day counting as customers enter the doors to regulate the flow of traffic.
## What it does: 
Through a connected wireless camera, the software tracks the number of people who enter and exit the building. Once a critical number of people have entered the room, the software will alert the user that this capacity has been exceeded.
## How we built it: 
Python, Anaconda, OpenCV, Tensorflow.
## Challenges we ran into: 
It was incredibly difficult to actually make the OpenCV model to track people in real time instead of just detecting their presence in a video frame. This kept on giving us errors such as there being thousands of people in a room at a time, when only one person was in front of the camera. We cycled through not only half a dozen computer vision models, but we were eventually able to settle on one that would perform well enough for our prototype.
## Accomplishments that we're proud of: 
The camera was working. It recognizes humans and it tracks the movement.
## What we learned: 
We learned about hardware components such as arduino OV7670, UNO R3. Then how to use anaconda and more of python, OpenCV, and tensorflow.
## What's next for Covid Cam TV: 
Building it in Tensorflow JS to not only make it compatible with more devices, but also run faster and on lighter systems for a smoother experience. We would also like to integrate more cameras and build a web app around it that would allow it to seamlessly integrate into the security system of the building.
