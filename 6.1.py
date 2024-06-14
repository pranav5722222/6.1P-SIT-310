import sys
import time

# Importing necessary libraries
import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CompressedImage

class LaneDetector:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("lane_detector_node")
        
        # Subscribing to the image topic
        self.image_sub = rospy.Subscriber('/duckie/camera_node/image/compressed', CompressedImage, self.process_image, queue_size=1)

    def process_image(self, msg):
        rospy.loginfo("Processing image")

        # Convert the compressed ROS image to an OpenCV image
        image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Define the cropping parameters
        top, bottom, left, right = 200, 400, 100, 500

        # Crop the image
        cropped_image = image[top:bottom, left:right]

        # Convert cropped image to HSV color space
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for white and yellow in HSV
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([255, 50, 255])
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([40, 255, 255])

        # Create masks for white and yellow colors
        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
        yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

        # Apply Canny Edge Detection to the masks
        white_edges = cv2.Canny(white_mask, 50, 150)
        yellow_edges = cv2.Canny(yellow_mask, 50, 150)

        # Apply Hough Transform to detect lines
        white_lines = self.detect_lines(white_edges)
        yellow_lines = self.detect_lines(yellow_edges)

        # Draw the detected lines on the cropped image
        self.draw_lines(cropped_image, white_lines)
        self.draw_lines(cropped_image, yellow_lines)

        # Display the processed masks
        cv2.imshow('White Mask', white_mask)
        cv2.imshow('Yellow Mask', yellow_mask)
        cv2.waitKey(1)

    def detect_lines(self, edge_img):
        # Apply the Hough Transform to detect lines
        lines = cv2.HoughLinesP(edge_img, 1, np.pi/180, 100, minLineLength=50, maxLineGap=50)
        return lines

    def draw_lines(self, image, lines):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        lane_detector = LaneDetector()
        lane_detector.run()
    except rospy.ROSInterruptException:
        pass
