// 2 Servo Motor Setup for Continuous and 180 deg servos


//Include the Servo Library
//Create a Servo Object

#include <Servo.h>
Servo francisServo;

//Define the Pins
int servoPin = 9;

void setup() {
  //"Attach" the servo object
  //We don't need to set it as an output manually
  francisServo.attach(servoPin);
  francisServo.write(90);
  delay(1000);
  //the process is like serial, because serial has a default library
  //Servo.h is a library also.
}

void loop() {
/*
  for (int i = 0; i <= 180; i = i + 20) {
    francisServo.write(i);
    delay(1000);
  }
*/
}
