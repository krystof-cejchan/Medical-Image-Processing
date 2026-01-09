import cv2 as cv
import numpy as np

def run_optical_flow(video_path):
    cap = cv.VideoCapture(video_path)

    feature_params = dict(maxCorners=50,
                          qualityLevel=0.3,
                          minDistance=20,
                          blockSize=5)
    
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    
    tracks = []
    
    detect_interval = 3
    frame_idx = 0
    
    ret, prev_frame = cap.read()
    if not ret:
        return
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vis = frame.copy()

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, _, _ = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, _, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            
   
            good = d < 1
            
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                    
                tr.append((x, y))
                
                if len(tr) > 20:
                    del tr[0]
                    
                new_tracks.append(tr)
                
                cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv.polylines(vis, [np.int32(tr)], False, (0, 255, 0), 1)
                
            tracks = new_tracks

        if frame_idx % detect_interval == 0:
            
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            
            for tr in tracks:
                x, y = tr[-1]
                cv.circle(mask, (int(x), int(y)), 5, 0, -1)
                
            p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])


        frame_idx += 1
        prev_gray = frame_gray
        
        cv.imshow('Sparse opt flow', vis)
        
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_optical_flow('./files/cars_moving2.mp4')