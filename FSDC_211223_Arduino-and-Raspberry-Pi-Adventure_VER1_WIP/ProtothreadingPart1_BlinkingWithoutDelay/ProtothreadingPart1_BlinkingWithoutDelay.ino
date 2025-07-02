const int ledPin = 13;

int ledState = LOW;
long previousMillis = 0;

long interval = 999;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {

  blinkFunction();


}

void blinkFunction() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis > interval) {
    Serial.print("Previous Millis: ");
    Serial.println(previousMillis);
    previousMillis = currentMillis;
    if (ledState == LOW) {
      ledState = HIGH;
    } else {
      ledState = LOW;
    }
    digitalWrite(ledPin, ledState);
    Serial.print("Current Millis: ");
    Serial.println(currentMillis);
  }

}
