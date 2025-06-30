#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

int rate = 500;
int angle0[] = {130, 120, 110, 100, 90};
int angle1[] = {80, 70, 60, 50, 90};
int angle2[] = {60, 50, 40, 30, 90};
int angle3[] = {60, 50, 40, 30, 90};
int angle4[] = {120, 110, 100, 90, 90};
int angle5[] = {140, 130, 120, 110, 90};
int angle6[] = {130, 120, 110, 100, 90};
int angle7[] = {80, 70, 60, 50, 90};
int angle8[] = {60, 50, 40, 30, 90};
int angle9[] = {60, 50, 40, 30, 90};
int angle10[] = {120, 110, 100, 90, 90};
int angle11[] = {140, 130, 120, 110, 90};


int pos1 = 0;
int pos2 = 0;
int pos3 = 0;
int pos4 = 0;
int pos5 = 0;
int pos6 = 0;
int pos7 = 0;
int pos8 = 0;
int pos9 = 0;
int pos10 = 0;
int pos11 = 0;
int pos12 = 0;
int updateInterval1;
int updateInterval2;
int updateInterval3;
int updateInterval4;
int updateInterval5;
int updateInterval6;
int updateInterval7;
int updateInterval8;
int updateInterval9;
int updateInterval10;
int updateInterval11;
int updateInterval12;
unsigned long lastUpdate1;
unsigned long lastUpdate2;
unsigned long lastUpdate3;
unsigned long lastUpdate4;
unsigned long lastUpdate5;
unsigned long lastUpdate6;
unsigned long lastUpdate7;
unsigned long lastUpdate8;
unsigned long lastUpdate9;
unsigned long lastUpdate10;
unsigned long lastUpdate11;
unsigned long lastUpdate12;

int arraySize = sizeof(angle2) / sizeof(angle2[0]);

void setup() {
  Serial.begin(9600);
  Serial.println("8 channel Servo test!");
  pwm.begin();
  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz updates
  delay(10);
  pwm.setPWM(0, 0, angleToPulse0(90));
  //pwm.setPWM(1, 0, angleToPulse1(90));
  pwm.setPWM(2, 0, angleToPulse2(90));
  pwm.setPWM(3, 0, angleToPulse3(90));
  //pwm.setPWM(4, 0, angleToPulse4(90));
  pwm.setPWM(5, 0, angleToPulse5(90));
  pwm.setPWM(6, 0, angleToPulse6(90));
  //pwm.setPWM(7, 0, angleToPulse7(90));
  pwm.setPWM(8, 0, angleToPulse8(90));
  pwm.setPWM(9, 0, angleToPulse9(90));
  //pwm.setPWM(10, 0, angleToPulse10(90));
  pwm.setPWM(11, 0, angleToPulse11(90));
  delay(1500);

  updateInterval1 = 500;
  updateInterval2 = 600;
  updateInterval3 = 700;
  updateInterval4 = 800;
  updateInterval5 = 900;
  updateInterval6 = 1000;
  updateInterval7 = 1100;
  updateInterval8 = 1200;
  updateInterval9 = 1300;
  updateInterval10 = 1400;
  updateInterval11 = 1500;
  updateInterval12 = 1600;

}

void loop() {
  if ((millis() - lastUpdate1) > updateInterval1) {
    lastUpdate1 = millis();
    pwm.setPWM(0, 0, angleToPulse0(angle0[pos1]));
    pos1++;
    if ((pos1 >= arraySize) || (pos1 <= 0)) {
      pos1 = 0;
    }
  }

  if ((millis() - lastUpdate2) > updateInterval2) {
    lastUpdate2 = millis();
    pwm.setPWM(1, 0, angleToPulse1(angle1[pos2]));
    pos2++;
    if ((pos2 >= arraySize) || (pos2 <= 0)) {
      pos2 = 0;
    }
  }

  if ((millis() - lastUpdate3) > updateInterval3) {
    lastUpdate3 = millis();
    pwm.setPWM(2, 0, angleToPulse2(angle2[pos3]));
    pos3++;
    if ((pos3 >= arraySize) || (pos3 <= 0)) {
      pos3 = 0;
    }
  }

  if ((millis() - lastUpdate4) > updateInterval4) {
    lastUpdate4 = millis();
    pwm.setPWM(3, 0, angleToPulse3(angle3[pos4]));
    pos4++;
    if ((pos4 >= arraySize) || (pos4 <= 0)) {
      pos4 = 0;
    }
  }

  if ((millis() - lastUpdate5) > updateInterval5) {
    lastUpdate5 = millis();
    pwm.setPWM(4, 0, angleToPulse4(angle4[pos5]));
    pos5++;
    if ((pos5 >= arraySize) || (pos5 <= 0)) {
      pos5 = 0;
    }
  }

  if ((millis() - lastUpdate6) > updateInterval6) {
    lastUpdate6 = millis();
    pwm.setPWM(5, 0, angleToPulse5(angle5[pos6]));
    pos6++;
    if ((pos6 >= arraySize) || (pos6 <= 0)) {
      pos6 = 0;
    }
  }


  if ((millis() - lastUpdate7) > updateInterval7) {
    lastUpdate7 = millis();
    pwm.setPWM(6, 0, angleToPulse6(angle6[pos7]));
    pos7++;
    if ((pos7 >= arraySize) || (pos7 <= 0)) {
      pos7 = 0;
    }
  }

  if ((millis() - lastUpdate8) > updateInterval8) {
    lastUpdate8 = millis();
    pwm.setPWM(7, 0, angleToPulse7(angle7[pos8]));
    pos8++;
    if ((pos8 >= arraySize) || (pos8 <= 0)) {
      pos8 = 0;
    }
  }

  if ((millis() - lastUpdate9) > updateInterval9) {
    lastUpdate9 = millis();
    pwm.setPWM(8, 0, angleToPulse8(angle8[pos9]));
    pos9++;
    if ((pos9 >= arraySize) || (pos9 <= 0)) {
      pos9 = 0;
    }
  }

  if ((millis() - lastUpdate10) > updateInterval10) {
    lastUpdate10 = millis();
    pwm.setPWM(9, 0, angleToPulse9(angle9[pos10]));
    pos10++;
    if ((pos10 >= arraySize) || (pos10 <= 0)) {
      pos10 = 0;
    }
  }

  if ((millis() - lastUpdate11) > updateInterval11) {
    lastUpdate11 = millis();
    pwm.setPWM(10, 0, angleToPulse10(angle10[pos11]));
    pos11++;
    if ((pos11 >= arraySize) || (pos11 <= 0)) {
      pos11 = 0;
    }
  }

  if ((millis() - lastUpdate12) > updateInterval12) {
    lastUpdate12 = millis();
    pwm.setPWM(11, 0, angleToPulse11(angle11[pos12]));
    pos12++;
    if ((pos12 >= arraySize) || (pos12 <= 0)) {
      pos12 = 0;
    }
  }
}

int angleToPulse0(int ang) {
  int pulse0 = map(ang, 0, 180, 160, 480);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse0);
  return pulse0;
}

int angleToPulse1(int ang) {
  int pulse1 = map(ang, 0, 180, 160, 490);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse1);
  return pulse1;
}

int angleToPulse2(int ang) {
  int pulse2 = map(ang, 0, 180, 160, 440);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse2);
  return pulse2;
}

int angleToPulse3(int ang) {
  int pulse3 = map(ang, 0, 180, 185, 500);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse3);
  return pulse3;
}
int angleToPulse4(int ang) {
  int pulse4 = map(ang, 0, 180, 190, 500);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse4);
  return pulse4;
}

int angleToPulse5(int ang) {
  int pulse5 = map(ang, 0, 180, 190, 490);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse5);
  return pulse5;
}

int angleToPulse6(int ang) {
  int pulse6 = map(ang, 0, 180, 160, 490);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse6);
  return pulse6;
}

int angleToPulse7(int ang) {
  int pulse7 = map(ang, 0, 180, 160, 480);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse7);
  return pulse7;
}

int angleToPulse8(int ang) {
  int pulse8 = map(ang, 0, 180, 175, 510);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse8);
  return pulse8;
}

int angleToPulse9(int ang) {
  int pulse9 = map(ang, 0, 180, 180, 500);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse9);
  return pulse9;
}

int angleToPulse10(int ang) {
  int pulse10 = map(ang, 0, 180, 160, 480);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse10);
  return pulse10;
}

int angleToPulse11(int ang) {
  int pulse11 = map(ang, 0, 180, 190, 510);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse11);
  return pulse11;
}
