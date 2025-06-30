#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <AFMotor.h>


// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
AF_Stepper motor(50, 1);

int rate = 500;
int angle0[] = {160, 120, 110, 100, 90};
int angle1[] = {140, 50, 140, 50, 90};
int angle2[] = {100, 50, 100, 50, 90};
int pos = 0;
int updateInterval;
unsigned long lastUpdate;

int arraySize = sizeof(angle2)/ sizeof(angle2[0]);

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
  updateInterval = 500;
}

void loop() {
  /*
    motor.step(50, FORWARD, DOUBLE);
    motor.release();
    delay(rate / 2);
    motor.step(50, BACKWARD, DOUBLE);
    motor.release();
    delay(rate / 2);
  */
  //Serial.println(arraySize);
  if ((millis() - lastUpdate) > updateInterval) {
    lastUpdate = millis();
    pwm.setPWM(0, 0, angleToPulse0(angle0[pos]));
    pwm.setPWM(1, 0, angleToPulse1(angle1[pos])); 
    pwm.setPWM(2, 0, angleToPulse2(angle2[pos]));
    pos++;
    Serial.println(pos);
    Serial.println(lastUpdate);
    if ((pos >= arraySize) || (pos <= 0)) {
      pos = 0;
    }
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
