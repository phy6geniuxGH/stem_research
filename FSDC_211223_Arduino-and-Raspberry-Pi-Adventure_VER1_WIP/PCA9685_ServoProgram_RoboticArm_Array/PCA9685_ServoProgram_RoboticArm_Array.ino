#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <AFMotor.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
AF_Stepper motor(50, 1);

int rate = 500;
int angle0[] = {140, 120, 110, 100, 90};
int angle1[] = {80, 70, 60, 50, 90};
int angle2[] = {60, 50, 40, 30, 90};


void setup() {
  Serial.begin(9600);
  Serial.println("8 channel Servo test!");
  motor.setSpeed(100); //10 rpm
  pwm.begin();
  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz updates
  delay(10);
  pwm.setPWM(0, 0, angleToPulse0(90));
  pwm.setPWM(1, 0, angleToPulse1(90));
  pwm.setPWM(2, 0, angleToPulse2(90));

  delay(1500);

}

void loop() {
  motor.step(50, FORWARD, DOUBLE);
  motor.release();
  delay(rate / 2);
  motor.step(50, BACKWARD, DOUBLE);
  motor.release();
  delay(rate / 2);
  for (int i = 0; i < 5; i++) {
    //pwm.setPWM(0, 0, angleToPulse0(angle0[i]));
    //pwm.setPWM(1, 0, angleToPulse1(angle1[i]));
    pwm.setPWM(2, 0, angleToPulse2(angle2[i]));
    delay(rate);
  }
}

int angleToPulse0(int ang) {
  int pulse0 = map(ang, 0, 180, 150, 570);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse0);
  return pulse0;
}

int angleToPulse1(int ang) {
  int pulse1 = map(ang, 0, 180, 150, 580);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse1);
  return pulse1;
}

int angleToPulse2(int ang) {
  int pulse2 = map(ang, 0, 180, 185, 520);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse2);
  return pulse2;
}
