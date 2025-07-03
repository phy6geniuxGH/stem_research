const int ledPin1 = 12;
int ledState1 = LOW;
unsigned long previousMillis1 = 0;
long OnTime1 = 700;
long OffTime1 = 400;

const int ledPin2 = 13;
int ledState2 = LOW;
unsigned long previousMillis2 = 0;
long OnTime2 = 1200;
long OffTime2 = 250;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin1, OUTPUT);
  pinMode(ledPin2, OUTPUT);
}

void loop() {

  blinkFunction();


}

void blinkFunction() {
  unsigned long currentMillis = millis();
  if ((ledState1 == HIGH) && (currentMillis - previousMillis1 >= OnTime1)) {
    ledState1 = LOW;
    previousMillis1 = currentMillis;
    digitalWrite(ledPin1, ledState1);
  }
  if ((ledState1 == LOW) && (currentMillis - previousMillis1 >= OffTime1)) {
    ledState1 = HIGH;
    previousMillis1 = currentMillis;
    digitalWrite(ledPin1, ledState1);
  }
  if ((ledState2 == HIGH) && (currentMillis - previousMillis2 >= OnTime2)) {
    ledState2 = LOW;
    previousMillis2 = currentMillis;
    digitalWrite(ledPin2, ledState2);
  }
  if ((ledState2 == LOW) && (currentMillis - previousMillis2 >= OffTime2)) {
    ledState2 = HIGH;
    previousMillis2 = currentMillis;
    digitalWrite(ledPin2, ledState2);
  }

}
