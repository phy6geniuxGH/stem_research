#include <Servo.h>

const int trigPin = 10;
const int echoPin = 11;

long duration;
int distance;

Servo francisServo;


void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
  francisServo.attach(9);

}
void loop() {

  for (int i = 15; i < 140; i++) {
    francisServo.write(i);
    delay(30);
    distance = calculateDistance();
    Serial.print(i);
    Serial.print(",");
    Serial.print(distance);
    Serial.print(".");
  }
  for (int i = 140; i > 15; i--) {
    francisServo.write(i);
    delay(30);
    distance = calculateDistance();
    Serial.print(i);
    Serial.print(",");
    Serial.print(distance);
    Serial.print(".");

  }

}
int calculateDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH); //Reads the echoPin, returns
  //sound wave travel time in microseconds.
  distance = duration * 0.034 / 2;
  // 2d = v_sound*time
  /*Calculation:
     => 2d[m]/t[s] = v_sound[m/s]
        2d[m]/t[s] * 1[s]/1000000[us] * 100[cm]/1[m] = v_sound[m/s]*1[s]/1000000[us] * 100[cm]/1[m]
        2d[cm]/t[us] = v_sound[cm/us]*100/1000000
        d[cm] = (100/2000000)*v_sound[cm/us]*t[us]

        v_sound = 340 m/s

        d[cm] = (34000/2000000)[cm/us]*t[us]
        d = 0.034*t/2
  */
  return distance;
}
