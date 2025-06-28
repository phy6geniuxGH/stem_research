#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

int rate = 500;

int angle4[] = {0, 5, 10,15,20,25,30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 175, 170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100,
                95, 90, 85, 80, 75, 70, 65, 60,55, 50, 45, 40, 35, 30,25,20,15,10,5,0};

int pos5 = 0;
int updateInterval5;
unsigned long lastUpdate5;
int scalar5 = 1;

int arraySize = sizeof(angle4) / sizeof(angle4[0]);

void setup() {
  Serial.begin(9600);
  Serial.println("8 channel Servo test!");
  pwm.begin();
  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz updates
  delay(10);
  pwm.setPWM(4, 0, angleToPulse4(30));
  delay(1500);

  updateInterval5 = 50;


}

void loop() {

  if ((millis() - lastUpdate5) > updateInterval5) {
    lastUpdate5 = millis();
    pwm.setPWM(4, 0, angleToPulse4(angle4[pos5]));
    pos5 += scalar5;
    if ((pos5 >= arraySize - 1) || (pos5 <= 0)) {
      scalar5 = -scalar5;
      //pos1 = 0;
    }
  }

}

/*__________________________________

   Calibration Code
   _________________________________
*/


int angleToPulse4(int ang) {
  int pulse4 = map(ang, 0, 180, 190, 500);
  Serial.print("Angle: "); Serial.print(ang);
  Serial.print(" pulse: "); Serial.println(pulse4);
  return pulse4;
}
