# -*- coding: utf-8 -*-
"""
Logitech USB Camera - Direct Index 0
"""

import cv2

# Langsung buka camera index 0 dengan DirectShow
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if cap.isOpened():
    # Set resolusi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera Index 0 opened successfully!")
    print("Press 'Q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Cannot read frame")
            break
        
        # Tampilkan info
        cv2.putText(frame, "Logitech Camera (Index 0) - Press Q to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Logitech USB Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Camera closed")
else:
    print("✗ Cannot open camera index 0")