import cv2

def open_computer_vision():
    
    print('''
        -- Opening webcam ---------
        Press 'q' to quit.
    ''')
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):

        ret,frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800,450))
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()