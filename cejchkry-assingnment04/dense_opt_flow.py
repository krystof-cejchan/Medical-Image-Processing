import cv2 as cv
import numpy as np

def dense_optical_flow(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, first_frame = cap.read()
    if not ret:
        return

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    
    hsv_mask = np.zeros_like(first_frame)
    hsv_mask[..., 1] = 255

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                            0.5, 5, 15, 1, 5, 1.2, 0)
        

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        _, motion_mask = cv.threshold(mag, 2.0, 255, cv.THRESH_BINARY)
        motion_mask = motion_mask.astype(np.uint8)

        k = np.ones((15, 30), np.uint8)
        
        
        processed_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, k, iterations = 9)


        contours, _ = cv.findContours(processed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        bbox_frame = frame.copy()
        
        for contour in contours:
            if cv.contourArea(contour) < 5000:
                continue
                
            x, y, w, h = cv.boundingRect(contour)
            
            cv.rectangle(bbox_frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
            

        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        hsv_mask[..., 2] = cv.normalize(processed_mask, None, 0, 255, cv.NORM_MINMAX)
        
        rgb_representation = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)


        scale = 0.4
        h, w = frame.shape[:2]
        dim = (int(w * scale), int(h * scale))
        
        res1 = cv.resize(bbox_frame, dim)
        res2 = cv.resize(rgb_representation, dim)
        
        cv.imshow("Dense Optical Flow", np.hstack([res1, res2]))
        
        prev_gray = gray
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    dense_optical_flow('files/truck_moving.mp4')