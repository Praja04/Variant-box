# -*- coding: utf-8 -*-
"""
Pallet Quality Control System
Laptop Camera + YOLO Detection + Validation
By: Yaser Ali Husen (Modified)
"""

import cv2
import numpy as np
from datetime import datetime
import openpyxl
import os
from ultralytics import YOLO
import yaml

# ============= CONFIGURATION =============
VARIANT_CONFIG = {
    'A': {
        'total_boxes': 12,
        'name': 'Variant A (3x4)',
        'grid': {'rows': 3, 'cols': 4}
    },
    'B': {
        'total_boxes': 20,
        'name': 'Variant B (4x5)',
        'grid': {'rows': 4, 'cols': 5}
    },
    'C': {
        'total_boxes': 14,
        'name': 'Variant C (2x7)',
        'grid': {'rows': 2, 'cols': 7}
    }
}

# Active variant (ganti sesuai kebutuhan: 'A', 'B', atau 'C')
CURRENT_VARIANT = 'A'

# YOLO Model Path
# Untuk testing, gunakan model pre-trained dulu
# Nanti ganti dengan model custom yang sudah di-train dengan gambar kardus
MODEL_PATH = 'yolov8n.pt'  # Atau 'kardus_model.pt' jika sudah punya

# Position tolerance (pixel) - seberapa jauh boleh meleset dari posisi ideal
POSITION_TOLERANCE = 80

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Excel output path
OUTPUT_PATH = "../output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============= GLOBAL VARIABLES =============
total_checks = 0
total_valid = 0
total_invalid = 0
last_result = None

# ============= EXCEL SETUP =============
excel_file = os.path.join(OUTPUT_PATH, f"pallet_qc_{datetime.now().strftime('%Y%m%d')}.xlsx")

if not os.path.exists(excel_file):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(("Timestamp", "Variant", "Expected", "Detected", "Status", "Errors"))
    wb.save(excel_file)

def log_to_excel(variant, expected, detected, status, errors):
    """Log validation result to Excel"""
    wb = openpyxl.load_workbook(excel_file)
    ws = wb.active
    ws.append((
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        variant,
        expected,
        detected,
        status,
        '; '.join(errors) if errors else 'OK'
    ))
    wb.save(excel_file)

def generate_expected_positions(variant, image_width, image_height):
    """
    Generate expected positions based on variant grid
    Positions are distributed evenly across the image
    """
    config = VARIANT_CONFIG[variant]
    rows = config['grid']['rows']
    cols = config['grid']['cols']
    
    # Calculate margins and spacing
    margin_x = int(image_width * 0.1)  # 10% margin
    margin_y = int(image_height * 0.1)
    
    usable_width = image_width - (2 * margin_x)
    usable_height = image_height - (2 * margin_y)
    
    spacing_x = usable_width / (cols - 1) if cols > 1 else 0
    spacing_y = usable_height / (rows - 1) if rows > 1 else 0
    
    positions = []
    for row in range(rows):
        for col in range(cols):
            x = int(margin_x + (col * spacing_x))
            y = int(margin_y + (row * spacing_y))
            positions.append({'x': x, 'y': y, 'row': row, 'col': col})
    
    return positions

def validate_detections(detections, expected_positions, variant):
    """
    Validate detected boxes against expected positions
    
    Returns:
        dict with validation results
    """
    config = VARIANT_CONFIG[variant]
    expected_count = config['total_boxes']
    detected_count = len(detections)
    
    errors = []
    matched_positions = []
    unmatched_detections = []
    
    # Check 1: Jumlah kardus
    if detected_count != expected_count:
        if detected_count < expected_count:
            errors.append(f"KURANG {expected_count - detected_count} kardus")
        else:
            errors.append(f"LEBIH {detected_count - expected_count} kardus")
    
    # Check 2: Posisi kardus
    used_positions = set()
    
    for det in detections:
        # Get center point of detected box
        center_x = int((det['x1'] + det['x2']) / 2)
        center_y = int((det['y1'] + det['y2']) / 2)
        
        # Find nearest expected position
        min_distance = float('inf')
        nearest_idx = None
        
        for idx, exp_pos in enumerate(expected_positions):
            if idx in used_positions:
                continue
            
            distance = np.sqrt((center_x - exp_pos['x'])**2 + (center_y - exp_pos['y'])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx
        
        # Check if within tolerance
        if nearest_idx is not None and min_distance <= POSITION_TOLERANCE:
            matched_positions.append({
                'position_idx': nearest_idx,
                'detected': {'x': center_x, 'y': center_y},
                'expected': expected_positions[nearest_idx],
                'distance': min_distance
            })
            used_positions.add(nearest_idx)
        else:
            unmatched_detections.append({
                'x': center_x, 
                'y': center_y,
                'nearest_distance': min_distance
            })
    
    # Check missing positions
    missing_positions = [i for i in range(len(expected_positions)) if i not in used_positions]
    
    if unmatched_detections:
        errors.append(f"{len(unmatched_detections)} kardus POSISI SALAH")
    
    if missing_positions:
        errors.append(f"{len(missing_positions)} posisi KOSONG")
    
    is_valid = len(errors) == 0
    
    return {
        'valid': is_valid,
        'errors': errors,
        'expected_count': expected_count,
        'detected_count': detected_count,
        'matched': len(matched_positions),
        'unmatched': len(unmatched_detections),
        'missing': len(missing_positions),
        'matched_details': matched_positions,
        'unmatched_details': unmatched_detections,
        'missing_positions': missing_positions
    }

def draw_overlay(frame, detections, expected_positions, validation_result, variant):
    """Draw detection results and validation status on frame"""
    frame_out = frame.copy()
    height, width = frame.shape[:2]
    
    # Draw expected positions (grid)
    for idx, pos in enumerate(expected_positions):
        color = (0, 255, 0) if idx not in validation_result['missing_positions'] else (0, 0, 255)
        cv2.circle(frame_out, (pos['x'], pos['y']), 5, color, -1)
        cv2.circle(frame_out, (pos['x'], pos['y']), POSITION_TOLERANCE, color, 1)
        cv2.putText(frame_out, str(idx+1), (pos['x']+10, pos['y']+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw detected boxes
    for det in detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Check if this detection is matched
        is_matched = any(m['detected']['x'] == center_x and m['detected']['y'] == center_y 
                        for m in validation_result['matched_details'])
        
        color = (0, 255, 0) if is_matched else (0, 165, 255)  # Green if matched, Orange if not
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame_out, (center_x, center_y), 3, color, -1)
        
        # Draw confidence
        conf_text = f"{det['confidence']:.2f}"
        cv2.putText(frame_out, conf_text, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw status panel
    panel_height = 180
    cv2.rectangle(frame_out, (10, 10), (width-10, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame_out, (10, 10), (width-10, panel_height), (255, 255, 255), 2)
    
    # Status color
    status_color = (0, 255, 0) if validation_result['valid'] else (0, 0, 255)
    status_text = "✓ VALID" if validation_result['valid'] else "✗ INVALID"
    
    y_offset = 35
    cv2.putText(frame_out, f"PALLET QC SYSTEM - {VARIANT_CONFIG[variant]['name']}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame_out, f"Status: {status_text}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    y_offset += 25
    cv2.putText(frame_out, f"Expected: {validation_result['expected_count']} | Detected: {validation_result['detected_count']}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(frame_out, f"Matched: {validation_result['matched']} | Wrong Pos: {validation_result['unmatched']} | Missing: {validation_result['missing']}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw errors
    if validation_result['errors']:
        y_offset += 25
        error_text = " | ".join(validation_result['errors'])
        cv2.putText(frame_out, f"Errors: {error_text}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw statistics
    y_offset += 30
    cv2.putText(frame_out, f"Total Checks: {total_checks} | Valid: {total_valid} | Invalid: {total_invalid}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Draw instructions
    instructions_y = height - 60
    cv2.rectangle(frame_out, (10, instructions_y-5), (width-10, height-10), (50, 50, 50), -1)
    cv2.putText(frame_out, "Controls: [SPACE] Check Pallet | [A/B/C] Switch Variant | [Q] Quit | [S] Save Screenshot", 
                (20, instructions_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame_out, f"Current Variant: [{variant}] - Press A, B, or C to change", 
                (20, instructions_y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return frame_out

# ============= MAIN PROGRAM =============
print("=" * 60)
print("PALLET QUALITY CONTROL SYSTEM")
print("=" * 60)
print(f"\nLoading YOLO model: {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nNote: Jika belum punya model YOLO, install dulu:")
    print("  pip install ultralytics")
    print("\nUntuk model custom kardus, train model dulu atau ganti MODEL_PATH")
    exit(1)

print(f"\nCurrent Variant: {VARIANT_CONFIG[CURRENT_VARIANT]['name']}")
print(f"Expected boxes: {VARIANT_CONFIG[CURRENT_VARIANT]['total_boxes']}")
print("\nOpening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Error: Cannot access webcam!")
    exit(1)

print("✓ Webcam opened successfully!")
print("\n" + "=" * 60)
print("READY! Press SPACE to check pallet, Q to quit")
print("=" * 60 + "\n")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error reading frame")
        break
    
    height, width = frame.shape[:2]
    
    # Generate expected positions for current variant
    expected_positions = generate_expected_positions(CURRENT_VARIANT, width, height)
    
    # If we have a last result, use it for display
    if last_result is not None:
        frame_display = draw_overlay(frame, last_result['detections'], 
                                     expected_positions, last_result['validation'], 
                                     CURRENT_VARIANT)
    else:
        # Just show guide
        frame_display = frame.copy()
        for idx, pos in enumerate(expected_positions):
            cv2.circle(frame_display, (pos['x'], pos['y']), 5, (100, 100, 100), -1)
            cv2.circle(frame_display, (pos['x'], pos['y']), POSITION_TOLERANCE, (100, 100, 100), 1)
        
        cv2.putText(frame_display, f"Variant {CURRENT_VARIANT} - Press SPACE to check", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow('Pallet QC System', frame_display)
    
    key = cv2.waitKey(1) & 0xFF
    
    # SPACE - Check pallet
    if key == ord(' '):
        print("\n" + "="*60)
        print(f"Checking pallet... Variant: {CURRENT_VARIANT}")
        
        # Run YOLO detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf,
                    'class': cls
                })
        
        # Validate
        validation_result = validate_detections(detections, expected_positions, CURRENT_VARIANT)
        
        # Update statistics
        total_checks += 1
        if validation_result['valid']:
            total_valid += 1
            print("✓ VALID - Pallet sesuai standard!")
        else:
            total_invalid += 1
            print("✗ INVALID - Pallet tidak sesuai!")
            print("Errors:")
            for error in validation_result['errors']:
                print(f"  - {error}")
        
        print(f"\nDetected: {validation_result['detected_count']}/{validation_result['expected_count']}")
        print(f"Matched: {validation_result['matched']}, Wrong Position: {validation_result['unmatched']}, Missing: {validation_result['missing']}")
        print("="*60)
        
        # Log to Excel
        log_to_excel(
            CURRENT_VARIANT,
            validation_result['expected_count'],
            validation_result['detected_count'],
            'VALID' if validation_result['valid'] else 'INVALID',
            validation_result['errors']
        )
        
        # Store result for display
        last_result = {
            'detections': detections,
            'validation': validation_result
        }
    
    # Switch variants
    elif key == ord('a') or key == ord('A'):
        CURRENT_VARIANT = 'A'
        last_result = None
        print(f"\n→ Switched to {VARIANT_CONFIG['A']['name']}")
    
    elif key == ord('b') or key == ord('B'):
        CURRENT_VARIANT = 'B'
        last_result = None
        print(f"\n→ Switched to {VARIANT_CONFIG['B']['name']}")
    
    elif key == ord('c') or key == ord('C'):
        CURRENT_VARIANT = 'C'
        last_result = None
        print(f"\n→ Switched to {VARIANT_CONFIG['C']['name']}")
    
    # Save screenshot
    elif key == ord('s') or key == ord('S'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_PATH, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame_display)
        print(f"\n📸 Screenshot saved: {filename}")
    
    # Quit
    elif key == ord('q') or key == ord('Q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total Checks: {total_checks}")
print(f"Valid: {total_valid} ({total_valid/total_checks*100:.1f}%)" if total_checks > 0 else "Valid: 0")
print(f"Invalid: {total_invalid} ({total_invalid/total_checks*100:.1f}%)" if total_checks > 0 else "Invalid: 0")
print(f"\nLog saved to: {excel_file}")
print("="*60)
print("\nProgram selesai. Terima kasih!")