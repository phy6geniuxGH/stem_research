//Servo + IR

#include <Servo.h>
Servo francisServo;

//Define the Pins
int servoPin = 9;
int distPin = 0;

void setup() {

  francisServo.attach(servoPin); 

}
void loop(){
  int dist = analogRead(distPin);
  int pos = map(dist, 160, 800, 180, 0);
  francisServo.write(pos);
}
