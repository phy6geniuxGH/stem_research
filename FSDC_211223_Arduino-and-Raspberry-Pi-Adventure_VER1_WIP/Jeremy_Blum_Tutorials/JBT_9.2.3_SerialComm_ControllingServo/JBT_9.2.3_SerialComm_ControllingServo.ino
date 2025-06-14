#include <Servo.h>

Servo francisSG90servo;

int ledPin = 13;
int servoPin = 9;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  francisSG90servo.attach(servoPin);
  francisSG90servo.write(90);
}

void loop() {
  // have the arduino wait to receive input
  while(Serial.available() == 0);
  //Read the Input
  int val = Serial.read() - '0';
  if (val == 1){
    Serial.println("LED is ON");
    digitalWrite(ledPin, HIGH);
    for(int i = 10; i <= 180; i++){
      francisSG90servo.write(i);
      delay(100);
    }
  } else if (val == 0){
    Serial.println("LED is OFF");
    digitalWrite(ledPin, LOW);
  } else {
    Serial.println("Invalid!");
  }
  delay(10);
  while(Serial.available()>0){
    Serial.read();
    delay(10);
  }
}

