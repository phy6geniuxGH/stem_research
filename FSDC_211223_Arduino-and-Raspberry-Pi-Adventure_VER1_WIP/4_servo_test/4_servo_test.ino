// 2 Servo Motor Setup for Continuous and 180 deg servos


//Include the Servo Library
//Create a Servo Object

#include <Servo.h>
Servo francisServo;
Servo francis2Servo;
Servo francis3Servo;
Servo francis4Servo;

//Define the Pins
int servoPin = 9;

void setup() {
  //"Attach" the servo object
  //We don't need to set it as an output manually
  francisServo.attach(10);
  francis2Servo.attach(11);
  francis3Servo.attach(6);
  francis4Servo.attach(5);
  francisServo.write(90);
  francis2Servo.write(90);
  francis3Servo.write(90);
  francis4Servo.write(90);
  delay(1000);
  //the process is like serial, because serial has a default library
  //Servo.h is a library also.
}

void loop() {

  for (int i = 50; i < 150; i = i + 50) {

    francis2Servo.write(i);
    francis3Servo.write(i);
    delay(250);
  }
  for (int i =150; i > 50; i = i - 50) {
    francisServo.write(i);
    francis4Servo.write(i);
    delay(250);
  }


}
