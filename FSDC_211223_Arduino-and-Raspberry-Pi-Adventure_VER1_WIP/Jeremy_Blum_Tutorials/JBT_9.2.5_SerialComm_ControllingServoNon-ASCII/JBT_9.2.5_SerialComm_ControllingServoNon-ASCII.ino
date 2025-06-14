//Servo Control using ASCII Data type values

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
  if (val < 90){
    Serial.println("LED is ON and Servo turns right");
    Serial.println(val, DEC); //converted to DEC
    digitalWrite(ledPin, HIGH);
    francisSG90servo.write(val);
  } else if (val >= 90){
    Serial.println("LED is OFF and Servo turns left");
    Serial.println(val, DEC);
    digitalWrite(ledPin, LOW);
    francisSG90servo.write(val);
  } else {
    Serial.println("Invalid!");
    Serial.println(val, DEC);
  }
  delay(10);
  while(Serial.available() > 0){
    Serial.read();
    delay(10);
  }
}

