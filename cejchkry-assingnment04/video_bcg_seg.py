import cv2 as cv
import numpy as np

def background_subtraction_comparison(video_path):
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("nešlo otevřít video")


    ret, first_frame = cap.read()
    if not ret:
        return
    
    accumulated_bg = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY).astype("float")
    

    alpha = 0.5
    diff_thresh_val = 10

    #mog2
    history = 10
    threshold = 40


    mog2_subtractor = cv.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold, detectShadows=False)


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Převod na odstíny šedi a rozmazání pro redukci šumu
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame_blur = cv.GaussianBlur(gray_frame, (5, 5), 0)


        cv.accumulateWeighted(gray_frame_blur, accumulated_bg, alpha)
        accumulated_bg_res = cv.convertScaleAbs(accumulated_bg)# Převod vypočteného pozadí zpět na 8-bitový integer pro zobrazení a odečítání
        
        diff_frame = cv.absdiff(gray_frame_blur, accumulated_bg_res)# Výpočet absolutního rozdílu mezi aktuálním snímkem a modelem pozadí
        
        _, accum_mask = cv.threshold(diff_frame, diff_thresh_val, 255, cv.THRESH_BINARY)# Prahování rozdílu -> vytvoření binární masky popředí (bílá = pohyb)

        mog2_mask = mog2_subtractor.apply(frame)# Metoda apply() automaticky aktualizuje model a vrátí masku popředí
        mog2_bg = mog2_subtractor.getBackgroundImage()

        scale = 0.6
        h, w = frame.shape[:2]
        dim = (int(w * scale), int(h * scale))
        
        original_resized = cv.resize(frame, dim)
        elem_base_size = 3
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * cv.MORPH_ELLIPSE + elem_base_size, 2 * cv.MORPH_ELLIPSE + elem_base_size),
                                       (cv.MORPH_ELLIPSE, cv.MORPH_ELLIPSE))
    

        dst = cv.morphologyEx(accum_mask, cv.MORPH_CLOSE, element, iterations = 4)
        
        accum_bg_display = cv.cvtColor(cv.resize(accumulated_bg_res, dim), cv.COLOR_GRAY2BGR)
        accum_fg_display = cv.cvtColor(cv.resize(accum_mask, dim), cv.COLOR_GRAY2BGR)

        if mog2_bg is not None:
            mog2_bg_display = cv.resize(mog2_bg, dim)
        else:
            mog2_bg_display = np.zeros_like(original_resized)
            
        mog2_fg_display = cv.cvtColor(cv.resize(mog2_mask, dim), cv.COLOR_GRAY2BGR)





        cv.putText(accum_bg_display, "Accum Weighted bcg", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(accum_fg_display, "Accum Weighted frg", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(mog2_bg_display, "MOG2 bcg", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(mog2_fg_display, "MOG2 fcg", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        row1 = np.hstack([accum_bg_display, accum_fg_display])
        row2 = np.hstack([mog2_bg_display, mog2_fg_display])
        
        cv.imshow('Original Frame', original_resized)
        cv.imshow('Comparison: Top=Accumulated, Bottom=MOG2', np.vstack([row1, row2]))

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    background_subtraction_comparison('files/video_pedestrians_movement.webm')
