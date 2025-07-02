const int ledPin = 13;

int ledState = LOW;
unsigned long previousMillis = 0;

long OnTime = 1000;
long OffTime = 50;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {

  blinkFunction();


}

void blinkFunction() {
  unsigned long currentMillis = millis();
  if ((ledState == HIGH) && (currentMillis - previousMillis >= OnTime)) {
    ledState = LOW;
    previousMillis = currentMillis;
    digitalWrite(ledPin, ledState);
  }
  if ((ledState == LOW) && (currentMillis - previousMillis >= OffTime)) {
    ledState = HIGH;
    previousMillis = currentMillis;
    digitalWrite(ledPin, ledState);
  }

}
