int ledPin = 13;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  // have the arduino wait to receive input
  while(Serial.available() == 0);
  //Read the Input
  int val = Serial.read() - '0';
  if (val == 1){
    Serial.println("LED is ON");
    digitalWrite(ledPin, HIGH);
  } else if (val == 0){
    Serial.println("LED is OFF");
    digitalWrite(ledPin, LOW);
  } else {
    Serial.println("Invalid!");
  }
}
