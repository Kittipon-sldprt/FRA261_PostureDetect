import cv2
import mediapipe as mp
import numpy as np
#This code can capture only posture from image (golf club not include) you need to draw a golf club and adjust by yourself
class StickmanDrawerWithCustomClubAngle:
    def __init__(self, static_image_mode=True, model_complexity=2):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode, 
            model_complexity=model_complexity, 
            enable_segmentation=False
        )
        self.pose_connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_ELBOW.value),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW.value, self.mp_pose.PoseLandmark.LEFT_WRIST.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value),
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value),
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value),
            (self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_HIP.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value),
        ]

    def _extract_keypoints(self, landmarks, image_shape):
        """Extract and scale pose landmarks to image dimensions."""
        h, w, _ = image_shape
        keypoints = {}
        for idx, landmark in enumerate(landmarks.landmark):
            keypoints[idx] = (int(landmark.x * w), int(landmark.y * h))
        return keypoints

    def _draw_stickman_with_club(self, image, keypoints, club_angle):
        """Draw the stickman and add a golf club with a specific angle."""
        # Create a blank white image
        stickman_image = np.ones_like(image) * 255

        # Draw connections
        for connection in self.pose_connections:
            pt1 = keypoints[connection[0]]
            pt2 = keypoints[connection[1]]
            cv2.line(stickman_image, pt1, pt2, (0, 0, 0), 2)

        # Draw keypoints as circles
        for point in keypoints.values():
            cv2.circle(stickman_image, point, 5, (0, 0, 255), -1)

        # Add a golf club
        if self.mp_pose.PoseLandmark.RIGHT_WRIST.value in keypoints:
            wrist = keypoints[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Define the endpoint of the club based on the custom angle

            #adjust length of golf club here
            club_length = 300  # Length of the golf club in pixels
            club_endpoint = (
                int(wrist[0] + club_length * np.cos(club_angle)),
                int(wrist[1] + club_length * np.sin(club_angle))
            )

            # Draw the club
            cv2.line(stickman_image, wrist, club_endpoint, (0, 255, 0), 3)

        return stickman_image

    def process_frame(self, image, club_angle=np.pi / 4):
        """
        Process an image to extract pose and draw a stickman with a golf club.
        :param club_angle: Angle of the club in radians relative to the horizontal axis.
        """
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process pose
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            keypoints = self._extract_keypoints(results.pose_landmarks, image.shape)
            stickman_image = self._draw_stickman_with_club(image, keypoints, club_angle)
            return stickman_image
        else:
            # Return a blank image if no pose is detected
            return np.ones_like(image) * 255

    def release(self):
        """Release resources."""
        self.pose.close()


# Example Usage
if __name__ == "__main__":
    # Initialize the stickman drawer
    stickman_drawer = StickmanDrawerWithCustomClubAngle()

    # Read an image
    image = cv2.imread('tiger_stand.png')  # Replace with your image path

    # Set custom club angle (e.g., 45 degrees, converted to radians)
    club_angle = np.deg2rad(90)  # Change this to set different angles

    # Process the image to draw a stickman with a custom club angle
    output_image = stickman_drawer.process_frame(image, club_angle=club_angle)

    # Show the stickman with the golf club
    cv2.imshow("Stickman with Golf Club (Custom Angle)", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release resources
    stickman_drawer.release()
