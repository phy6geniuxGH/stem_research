#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  350 // this is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  500// this is the 'maximum' pulse length count (out of 4096)

int rate = 250;

void setup() {
  Serial.begin(9600);
  Serial.println("8 channel Servo test!");

  pwm.begin();

  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz updates

  delay(10);


}

void loop() {
  frontLeftLeg();
  frontRightLeg();
}

void frontLeftLeg() {

  for (uint16_t pulselen = 320; pulselen < 400; pulselen++) {
    pwm.setPWM(0, 0, pulselen);
    pwm.setPWM(6, 0, pulselen);
  }
  delay(rate);


  for (uint16_t pulselen = 300; pulselen > 0; pulselen--) {
    pwm.setPWM(1, 0, pulselen);
    pwm.setPWM(7, 0, pulselen);
  }
  delay(rate);


  for (uint16_t pulselen = 400; pulselen > 320; pulselen--) {
    pwm.setPWM(0, 0, pulselen);
    pwm.setPWM(6, 0, pulselen);
  }
  delay(rate);


  for (uint16_t pulselen = 0; pulselen < 300; pulselen++) {
    pwm.setPWM(1, 0, pulselen);
    pwm.setPWM(7, 0, pulselen);
  }
  delay(rate);


  for (uint16_t pulselen = 290; pulselen > 190; pulselen--) {
    pwm.setPWM(2, 0, pulselen);
    pwm.setPWM(8, 0, pulselen);
  }
  delay(rate);


  for (uint16_t pulselen = 190; pulselen < 290; pulselen++) {
    pwm.setPWM(2, 0, pulselen);
    pwm.setPWM(8, 0, pulselen);
  }
  delay(rate);
}

void frontRightLeg() {
  for (uint16_t pulselen = 320; pulselen > 240; pulselen--) {
    pwm.setPWM(3, 0, pulselen);
    pwm.setPWM(9, 0, pulselen);
  }
  delay(rate);


  for (uint16_t pulselen = 300; pulselen < 600; pulselen++) {
    pwm.setPWM(4, 0, pulselen);
    pwm.setPWM(10, 0, pulselen);
  }
  delay(rate);

  for (uint16_t pulselen = 240; pulselen < 320; pulselen++) {
    pwm.setPWM(3, 0, pulselen);
    pwm.setPWM(9, 0, pulselen);
  }
  delay(rate);

  for (uint16_t pulselen = 600; pulselen > 300; pulselen--) {
    pwm.setPWM(4, 0, pulselen);
    pwm.setPWM(10, 0, pulselen);
  }
  delay(rate);

  for (uint16_t pulselen = 320; pulselen < 405; pulselen++) {
    pwm.setPWM(5, 0, pulselen);
    pwm.setPWM(11, 0, pulselen);
  }
  delay(rate);

  for (uint16_t pulselen = 405; pulselen > 320; pulselen--) {
    pwm.setPWM(5, 0, pulselen);
    pwm.setPWM(11, 0, pulselen);
  }
  delay(rate);



}
