#include <AFMotor.h>


AF_Stepper motor(50, 1);
void setup() {
  Serial.begin(9600);
  Serial.println("Stepper test!");
  motor.setSpeed(100); //10 rpm
}

void loop() {
  Serial.println("Single coil steps");
  motor.step(200, FORWARD, DOUBLE);
  motor.release();
  delay(250);
  motor.step(200, BACKWARD, DOUBLE);
  motor.release();
  delay(250);

}
