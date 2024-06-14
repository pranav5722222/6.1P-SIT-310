import sys
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np

class LaneDetectionNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("lane_detection_node")
        self.bridge = CvBridge()

        # Subscribe to the compressed image topic
        self.image_sub = rospy.Subscriber('/duckie/camera_node/image/compressed', CompressedImage, self.process_image, queue_size=1)

    def process_image(self, msg):
        rospy.loginfo("Image received")

        # Convert the compressed image message to an OpenCV image
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Define the region of interest for cropping
        top, bottom, left, right = 100, 400, 100, 600

        # Crop the image
        cropped_frame = frame[top:bottom, left:right]

        # Convert the cropped image to grayscale
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Set thresholds for detecting white lane markings
        lower_white_threshold = 200
        upper_white_threshold = 255

        # Apply threshold to get only white pixels
        white_mask = cv2.inRange(gray_frame, lower_white_threshold, upper_white_threshold)

        # Apply Canny Edge Detection
        edge_detected = cv2.Canny(white_mask, 50, 150)

        # Use Hough Transform to detect lines
        lines_detected = cv2.HoughLinesP(edge_detected, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)

        # Draw detected lines on the original image
        if lines_detected is not None:
            for line in lines_detected:
                x1, y1, x2, y2 = line[0]
                cv2.line(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the image with detected lines
        cv2.imshow('Lane Lines', cropped_frame)
        cv2.waitKey(1)

    def start(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        lane_detection_node = LaneDetectionNode()
        lane_detection_node.start()
    except rospy.ROSInterruptException:
        pass
