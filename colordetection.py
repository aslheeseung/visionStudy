import numpy as np
import cv2


class ColorDetector():

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(0)
    
    def detect_color_hsv(self, frame, color):
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        if color == 'red':
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            mask = mask1 + mask2
        elif color == 'blue':
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
            mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        elif color == 'yellow':
            lower_yellow = np.array([25, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        else:
            return None

        return mask, cv2.bitwise_and(frame, frame, mask=mask)
    

    def draw_top_contours(self, frame, mask, min_area=100, top=3):
        # 모폴로지 연산 적용
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        large_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)[:top]

        for contour in large_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame
    

    def process_frame(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            for color in ['red', 'blue', 'yellow']:
                mask, detected = self.detect_color_hsv(frame, color)
                if detected is not None:
                    frame = self.draw_top_contours(frame, mask,3)

            cv2.imshow('Color Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

detector = ColorDetector()
detector.process_frame()