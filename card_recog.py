from ultralytics import YOLO
import cv2
import time
import numpy as np

# Import the blackjack advice function
from advice import blackjack_advice

# Card value mapping (converting card names to numerical values)
def card_to_value(card_name):
    # Handle formats like "10D", "KH", "AS", etc.
    # First character(s) is the rank, last character is the suit
    
    # Remove the suit (last character)
    rank = card_name[:-1].upper()
    
    # Map the rank to a numerical value
    if rank == 'A':
        return 11  # Ace is counted as 11 initially
    elif rank in ['J', 'Q', 'K']:
        return 10
    elif rank.isdigit():
        return int(rank)
    
    # If we can't parse it, print for debugging and return 0
    print(f"Warning: Couldn't parse card value from '{card_name}'")
    return 0

# Load the YOLO model for playing card detection
model = YOLO("yolov8s_playing_cards.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1102)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)

print("Starting Blackjack Advisor. Press 'q' to quit.")
print("Place dealer's card in the top half of the screen.")
print("Place player's cards in the bottom half of the screen.")

# Function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    if union_area > 0:
        return intersection_area / union_area
    else:
        return 0

# Function to filter duplicate cards
def filter_unique_cards(card_detections, distance_threshold=100, iou_threshold=0.2):
    if not card_detections:
        return []
    
    # Sort by confidence score (highest first)
    sorted_cards = sorted(card_detections, key=lambda x: x[2], reverse=True)
    
    unique_cards = []
    used_card_types = set()
    
    for card in sorted_cards:
        card_value, card_name, confidence, bbox = card
        
        # Skip this card if we already have one with the same name
        if card_name in used_card_types:
            continue
        
        # Check if this card is too close to any existing card (regardless of name)
        is_duplicate = False
        for unique_card in unique_cards:
            unique_bbox = unique_card[3]
            
            # Calculate IoU - if high enough, consider duplicate
            iou = calculate_iou(bbox, unique_bbox)
            if iou > iou_threshold:
                is_duplicate = True
                break
            
            # Also check centers distance as another measure to catch near duplicates
            center1 = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            center2 = ((unique_bbox[0] + unique_bbox[2]) / 2, (unique_bbox[1] + unique_bbox[3]) / 2)
            
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            if distance < distance_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_cards.append(card)
            used_card_types.add(card_name)
    
    return unique_cards

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Draw a line separating top and bottom halves
    cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 0), 2)
    
    # Run inference on the whole frame
    results = model(frame)
    
    # Initialize lists for dealer and player cards
    dealer_cards = []
    player_cards = []
    
    # Process results
    for result in results:
        for box in result.boxes:
            class_id = box.cls.item()  # Class ID
            class_name = result.names[class_id]  # Class name (e.g., "10D", "KH")
            confidence = box.conf.item()  # Confidence score
            
            # Skip detections with low confidence
            if confidence < 0.5:
                continue
                
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Calculate center point of the bounding box
            center_y = (y1 + y2) // 2
            
            # Determine if the card is in the top or bottom half
            card_value = card_to_value(class_name)
            if center_y < height // 2:
                # Dealer card (top half)
                dealer_cards.append((card_value, class_name, confidence, (x1, y1, x2, y2)))
            else:
                # Player card (bottom half)
                player_cards.append((card_value, class_name, confidence, (x1, y1, x2, y2)))
    
    # Apply strong filtering to remove duplicates
    dealer_cards = filter_unique_cards(dealer_cards)
    player_cards = filter_unique_cards(player_cards)
    
    # Draw detections on the frame
    annotated_frame = results[0].plot()
    
    # Draw a line separating top and bottom halves (again, as plot() may have overwritten it)
    cv2.line(annotated_frame, (0, height//2), (width, height//2), (0, 255, 0), 2)
    
    # Draw bounding boxes for the filtered cards with unique colors to verify
    for i, card in enumerate(dealer_cards):
        x1, y1, x2, y2 = card[3]
        color = (0, 255, 0)  # Green for dealer's cards
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    for i, card in enumerate(player_cards):
        x1, y1, x2, y2 = card[3]
        color = (0, 0, 255)  # Red for player's cards
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Add labels to the areas
    cv2.putText(annotated_frame, "Dealer's Card", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, "Player's Cards", (10, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Process detected cards for blackjack advice
    if dealer_cards and player_cards:
        # Get the dealer's card
        dealer_card_value = dealer_cards[0][0]  # Just the numerical value
        dealer_card_name = dealer_cards[0][1]
        
        # Get player's hand values
        player_hand_values = [card[0] for card in player_cards]
        player_card_names = [card[1] for card in player_cards]
        
        # For debugging: print the actual detected cards
        print(f"Dealer cards: {[c[1] for c in dealer_cards]}")
        print(f"Player cards: {[c[1] for c in player_cards]}")
        
        # Get advice using the advice function
        advice = blackjack_advice(player_hand_values, dealer_card_value)
        
        # Display the cards and advice
        dealer_text = f"Dealer: {dealer_card_name} (Value: {dealer_card_value})"
        player_text = f"Player: {', '.join(player_card_names)} (Sum: {sum(player_hand_values)})"
        
        # Display the advice in a colored box
        advice_text = f"Advice: {advice}"
        
        # Use green for blackjack, yellow for other advice
        advice_color = (0, 255, 0) if advice == "Blackjack!" else (0, 255, 255)
        
        cv2.rectangle(annotated_frame, (width//2 - 150, height - 100), (width//2 + 150, height - 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, advice_text, (width//2 - 140, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, advice_color, 2)
        
        # Display card information
        cv2.putText(annotated_frame, dealer_text, (10, height//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, player_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show the frame with detections
    cv2.imshow("Blackjack Advisor", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Add a small delay to reduce CPU usage
    time.sleep(0.01)

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()