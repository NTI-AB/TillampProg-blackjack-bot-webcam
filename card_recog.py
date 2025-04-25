from ultralytics import YOLO
import cv2
import time
import numpy as np

# Importera vår 5-stjärniga blackjack coach
from advice import blackjack_advice

# Den här skiten försöker översätta kort som "10D", "KH", "AS" till siffror vi kan jobba med
def card_to_value(card_name):
    # Vi skiter i färgen, bara siffran eller bokstaven spelar roll
    rank = card_name[:-1].upper()

    # Gör om bokstäver till värden, duh
    if rank == 'A':
        return 11  # Äss är alltid drama – räknas som 11
    elif rank in ['J', 'Q', 'K']:
        return 10  # Royals är lata, alltid 10
    elif rank.isdigit():
        return int(rank)  # Allt annat = ba en siffra

    # Om inget funkar så ba "no clue, bro"
    print(f"Warning: Couldn't parse card value from '{card_name}'")
    return 0

# Ladda in YOLO som ska leka kortdetektiv
model = YOLO("yolov8s_playing_cards.pt")

# Starta kameran, hoppas den inte suger
cap = cv2.VideoCapture(1)

# Verkar som kameran dog innan vi ens började
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Tvinga kameran till lite anständig upplösning
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1102)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)

print("Blackjack Advisor är igång. Tryck 'q' för att ge upp.")
print("Släng dealerns kort i övre halvan, dina kort i den undre.")

# Den här gör matte på lådor. Kollar hur mycket två boxar överlappar.
def calculate_iou(box1, box2):
    # Plocka ut koordinaterna
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Kolla var boxarna möts
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Om de inte nuddar varann alls
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Total yta minus överlapp = union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

# Tar bort dubbelkort eftersom YOLO ibland ser i syne
def filter_unique_cards(card_detections, distance_threshold=100, iou_threshold=0.2):
    if not card_detections:
        return []

    # Sortera efter YOLO:s självförtroende
    sorted_cards = sorted(card_detections, key=lambda x: x[2], reverse=True)
    
    unique_cards = []
    used_card_types = set()

    for card in sorted_cards:
        card_value, card_name, confidence, bbox = card

        # Redan sett detta kort? Vidare.
        if card_name in used_card_types:
            continue

        is_duplicate = False
        for unique_card in unique_cards:
            unique_bbox = unique_card[3]

            # Jämför boxarna. Om de nästan är samma, så skippa.
            iou = calculate_iou(bbox, unique_bbox)
            if iou > iou_threshold:
                is_duplicate = True
                break

            # Alternativt, om de typ sitter ihop
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
    # Få en frame från kameran
    ret, frame = cap.read()

    if not ret:
        print("Error: Kameran har gett upp.")
        break

    height, width = frame.shape[:2]

    # Dra en cool grön linje för att visa "här går gränsen"
    cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 0), 2)

    # YOLO, gör din grej
    results = model(frame)

    dealer_cards = []
    player_cards = []

    for result in results:
        for box in result.boxes:
            class_id = box.cls.item()
            class_name = result.names[class_id]
            confidence = box.conf.item()

            if confidence < 0.5:
                continue  # YOLO är osäker, vi skiter i den

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_y = (y1 + y2) // 2

            card_value = card_to_value(class_name)
            if center_y < height // 2:
                dealer_cards.append((card_value, class_name, confidence, (x1, y1, x2, y2)))
            else:
                player_cards.append((card_value, class_name, confidence, (x1, y1, x2, y2)))

    # Rensa bort YOLO:s hallucinationer
    dealer_cards = filter_unique_cards(dealer_cards)
    player_cards = filter_unique_cards(player_cards)

    annotated_frame = results[0].plot()

    # YOLO kanske ritat över linjen, så vi gör det igen lol
    cv2.line(annotated_frame, (0, height//2), (width, height//2), (0, 255, 0), 2)

    # Rita boxar – grönt = dealer, rött = du
    for i, card in enumerate(dealer_cards):
        x1, y1, x2, y2 = card[3]
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    for i, card in enumerate(player_cards):
        x1, y1, x2, y2 = card[3]
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Lite text så man vet vad som är vad
    cv2.putText(annotated_frame, "Dealer's Card", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, "Player's Cards", (10, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if dealer_cards and player_cards:
        dealer_card_value = dealer_cards[0][0]
        dealer_card_name = dealer_cards[0][1]
        player_hand_values = [card[0] for card in player_cards]
        player_card_names = [card[1] for card in player_cards]

        print(f"Dealer cards: {[c[1] for c in dealer_cards]}")
        print(f"Player cards: {[c[1] for c in player_cards]}")

        advice = blackjack_advice(player_hand_values, dealer_card_value)

        dealer_text = f"Dealer: {dealer_card_name} (Value: {dealer_card_value})"
        player_text = f"Player: {', '.join(player_card_names)} (Sum: {sum(player_hand_values)})"
        advice_text = f"Advice: {advice}"

        advice_color = (0, 255, 0) if advice == "Blackjack!" else (0, 255, 255)

        cv2.rectangle(annotated_frame, (width//2 - 150, height - 100), (width//2 + 150, height - 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, advice_text, (width//2 - 140, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, advice_color, 2)
        cv2.putText(annotated_frame, dealer_text, (10, height//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, player_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Blackjack Advisor", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

# Stäng allt innan datorn exploderar
cap.release()
cv2.destroyAllWindows()
