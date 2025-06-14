//Servo + IR

#include <Servo.h>
Servo francisServo;

//Define the Pins
int servoPin = 9;
int sensorPin = 0;
int sensorValue = 0;
int sensorMin = 6;
int sensorMax = 506;

void setup() {
  Serial.begin(9600);
  while (millis() < 10000){
    sensorValue = analogRead(sensorPin);
    if(sensorValue > sensorMax){
      sensorMax = sensorValue;
    }
    if(sensorValue < sensorMin){
      sensorMin = sensorValue;
    }
    Serial.println(sensorMin);
    Serial.println(sensorMax);
    delay(10);
    francisServo.attach(servoPin); 
  }
}
void loop(){
  sensorValue = analogRead(sensorPin);
  sensorValue = map(sensorValue, sensorMin, sensorMax, 180, 0); 
  sensorValue = constrain(sensorValue, 0, 180);
  Serial.println(sensorValue);
  delay(100);
  francisServo.write(sensorValue);
}
