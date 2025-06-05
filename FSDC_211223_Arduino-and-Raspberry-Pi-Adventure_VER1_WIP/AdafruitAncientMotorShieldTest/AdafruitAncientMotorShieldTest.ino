#include <AFMotor.h>

#define button 2
int rotDirection = 0;
boolean pressed = false;
AF_DCMotor motor1(1);
void setup() {
}

void loop() {
  int potValue = analogRead(A0);
  int pmwOutput = map(potValue, 0, 1023,0,255);
  motor1.setSpeed(pmwOutput);

  if (digitalRead(button) == true){
    pressed = !pressed;
  }
  while (digitalRead(button) == true){
    delay(20);
  }
  if (pressed == true && rotDirection == 0){
    motor1.run(FORWARD);
    rotDirection = 1;
    delay(20);
  }
    if (pressed == false && rotDirection == 1){
    motor1.run(BACKWARD);
    rotDirection = 0;
    delay(20);
  }
}
