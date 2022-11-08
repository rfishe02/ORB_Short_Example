import cv2

# GETTING THE KEY POINTS

# To identify something, we need an example of what it could look like
known_image = cv2.imread('./images/chip1.png')

# Convert the image from BGR to RGB (necessary for OpenCV), then convert to gray scale to improve outcome (could test this)
# We call this the training image, the image the algorithm will "learn" to recognize
training_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

# Load an image of something we want to identify
unknown_image = cv2.imread('./images/chip2.png')

# Make the same conversions as earlier, this could be considered "normalization", & it should be consistent for all images put into the algorithm
# This is the image for our "test"
test_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

# Create the object that can perform ORB
orb = cv2.ORB_create()

# Pass our images into the detectAndCompute method
# We get training keypoints & training descriptors in return
training_keypoints, training_descriptor = orb.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

# FIND MATCHING KEY POINTS

# Create a Brute Force Matcher object
bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck = True)

# Perform the matching between the ORB descriptors of the training image and the test image
matches = bf.match(training_descriptor, test_descriptor)

# See how many key points match (optional)
print(len(matches))
