int ledPin[] = {12, 13};
int ledState[] = {LOW, LOW};
unsigned long previousMillis[] = {0, 0};
long OnTime[] = {700, 1200};
long OffTime[] = {400, 250};

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 2; i++) {
    pinMode(ledPin[i], OUTPUT);
  }
}

void loop() {

  blinkFunction();


}

void blinkFunction() {
  unsigned long currentMillis = millis();
  for (int i = 0; i < 2; i++) {
    if ((ledState[i] == HIGH) && (currentMillis - previousMillis[i] >= OnTime[i])) {
      ledState[i] = LOW;
      previousMillis[i] = currentMillis;
      digitalWrite(ledPin[i], ledState[i]);
    }
    if ((ledState[i] == LOW) && (currentMillis - previousMillis[i] >= OffTime[i])) {
      ledState[i] = HIGH;
      previousMillis[i] = currentMillis;
      digitalWrite(ledPin[i], ledState[i]);
    }
  }
}
