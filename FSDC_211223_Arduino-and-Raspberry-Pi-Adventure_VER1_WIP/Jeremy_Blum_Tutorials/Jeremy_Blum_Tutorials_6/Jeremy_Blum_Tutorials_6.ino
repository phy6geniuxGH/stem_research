//Photoresistor as switch!
int sensePin = 0;
int ledPin = 9;

void setup() {
  analogReference(DEFAULT);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int val = analogRead(sensePin);
  if(val <= 800) digitalWrite(ledPin, HIGH);
  else digitalWrite(ledPin, LOW);
}
