//Servo Motor

//Include the Servo Library
//Create a Servo Object

#include <Servo.h>
Servo francisServo1;
Servo francisServo2;

//Define the Pins
int servoPin1 = 9;
int servoPin2 = 10;

void setup() {
  //"Attach" the servo object
  //We don't need to set it as an output manually
  francisServo1.attach(servoPin1); 
  francisServo2.attach(servoPin2);
  //the process is like serial, because serial has a default library
  //Servo.h is a library also.

}

void loop() {

  
  //Let's turn the servo from 0 to 180
  //in increments of 20 degrees
  for(int i=0; i <=90; i=i+2){
    francisServo1.write(i);
    delay(100);
  }
  
  for(int i=10; i<= 180; i=i+2){
    francisServo2.write(i);
    delay(10);
  }
  
}
