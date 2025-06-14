//Servo + IR

#include <Servo.h>
Servo francisServo;

//Define the Pins
int servoPin = 9;
int sensorPin = 0;
int sensorValue = 0;
int sensorMin = 6;
int sensorMax = 506;
int lastDist = 0;
int currentDist = 0;
int thresh = 10;
int motorPin11 = 11;

void setup() {
    adjustSensor();
    francisServo.attach(servoPin);
    pinMode(motorPin11, OUTPUT);
    
}
void loop(){
  calibratingSensor();
  currentDist = sensorValue;
  if(currentDist > lastDist + thresh || currentDist < lastDist - thresh){
    francisServo.write(sensorValue);
    analogWrite(motorPin11, sensorValue);
  lastDist = currentDist;
  }
  
  
}
void adjustSensor(){
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
  }
}
void calibratingSensor(){
  sensorValue = analogRead(sensorPin);
  sensorValue = map(sensorValue, sensorMin, sensorMax, 180, 0); 
  sensorValue = constrain(sensorValue, 0, 255);
  Serial.println(sensorValue);
  delay(100);
}

