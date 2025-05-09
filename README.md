# Blackjack Advisor med YOLOv8

## 🃏 Beskrivning
Detta projekt använder datorseende (YOLOv8), en webbkamera och en gnutta spelsmart AI för att analysera spelkort i realtid och ge råd i Blackjack. Spelaren lägger sina kort i nedre delen av kamerabilden, dealerns kort hamnar i övre – resten sköts automatiskt. Perfekt för den som vill fuska med stil, eh... jag menar, *träna strategiskt*.

---

## 🎥 Demonstration
[![YouTube Video](https://img.youtube.com/vi/qFE6WxIe4CM/maxresdefault.jpg)](https://youtube.com/shorts/qFE6WxIe4CM?feature=share)

---

## ⚙️ Funktioner
- Identifierar spelkort med hjälp av YOLOv8 och tränad modell (`playing_cards.pt`).
- Dela upp kamerabilden i dealer- och spelarzon för smart analys.
- Filtrerar dubbletter och hallucinationer (YOLOs livliga fantasi).
- Räknar ihop värden och ger råd enligt Blackjack-strategi.
- Fungerar i realtid via vanlig USB-kamera.

---

## 🧩 Komponenter
- YOLOv8-modell tränad på spelkort
- Python (3.8+)
- OpenCV
- Ultralytics YOLOv8
- En kamera som inte suger (USB rekommenderas)
- advice.py med strategi för Blackjack

---

## 🚀 Installation och Användning

### 1. Installera beroenden
```bash
pip install ultralytics opencv-python numpy
```

### 2. Klona repot
```bash
git clone https://github.com/ditt-användarnamn/blackjack-advisor.git
cd blackjack-advisor
```

### 3. Ladda ner modellen
Placera `yolov8s_playing_cards.pt` i projektmappen. (Finns inte här? Träna din egen eller fråga ChatGPT.)

### 4. Kör
```bash
python blackjack_camera.py
```

Placera **dealerns kort i övre halvan** och **spelarens kort i nedre**, sen är det bara att njuta.

## 🧠 Kodöversikt
```python
model = YOLO("yolov8s_playing_cards.pt")
cap = cv2.VideoCapture(1)  # Välj rätt kamera!
results = model(frame)
advice = blackjack_advice(player_hand, dealer_card)
```

## 💡 Utökningar
* Träna egen YOLO-modell på snyggare kort
* GUI med Tkinter eller PyQt  
* Raspberry Pi-stöd för portabelt fuskverktyg™
* Röststyrning ("Hey Siri, ska jag stanna?")

## 📜 Licens
Asså det här är ett skolprojekt lol, ingen aning vad för licenser som använts.
