//Robot car with IR and Servo Control

#include <Servo.h>
#define enA 9
#define in1 4
#define in2 5
#define enB 10
#define in3 6
#define in4 7
#define buttonPin1 11
#define buttonPin2 3
#define switchPin 2

int motorSpeedA = 0;
int motorSpeedB = 0;
boolean buttonState1 = LOW;
boolean buttonState2 = LOW;

int ledPin13 = 13;

Servo francisServo;
int servoPin = 12;
int sensorPin = A2;
int sensorValue = 0;
int sensorMin = 6;
int sensorMax = 506;
int lastDist = 0;
int currentDist = 0;
int thresh = 50;

int STATE = 0;
int pressed = false;

void setup(){
  francisServo.attach(servoPin);
  servoCenter();
  adjustSensor();
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(buttonPin1, OUTPUT);
  pinMode(buttonPin2, OUTPUT);
  pinMode(switchPin, INPUT);
  pinMode(ledPin13, OUTPUT);
  delay(50);
}
void loop(){
  setupSensor();
  int joyStickSwitch = digitalRead(switchPin);
  if (joyStickSwitch == true){
    pressed = !pressed; 
  }
  if(pressed == true){
    digitalWrite(ledPin13, HIGH);
    robotManualControl();
  }
  if(pressed == false){
    digitalWrite(ledPin13, LOW);
    IRcontrolLeftRight();
  }
}

void servoCenter(){
  for (int i = 0 ; i <=180 ; i = 90){
    francisServo.write(90);
    delay(100);
    break;
  }
}

void adjustSensor(){
  Serial.begin(9600);
  while (millis() < 10000){
    sensorValue = analogRead(sensorPin);
    if(sensorValue > sensorMax){
      sensorMax = sensorValue;
    }
    if(sensorValue < sensorMin){
      sensorMin = sensorValue;
    }
    Serial.println(sensorMin);
    Serial.println(sensorMax);
    delay(10);
  }
}
void calibratingSensor(){
  sensorValue = analogRead(sensorPin);
  sensorValue = map(sensorValue, sensorMin, sensorMax, 180, 0); 
  sensorValue = constrain(sensorValue, 0, 255);
  Serial.println(sensorValue);
  delay(100);
}
void setupSensor(){
  calibratingSensor();
  currentDist = sensorValue;
  if(currentDist > lastDist + thresh || currentDist < lastDist - thresh){
    francisServo.write(sensorValue);
    //analogWrite(motorPin11, sensorValue);
  lastDist = currentDist;
  }
}

void robotManualControl(){
  int xAxis = analogRead(A0);
  int yAxis = analogRead(A1);
  //Y-axis for forward and backward control (0 - 1023, 470 - 550 as center threshold)
  if (yAxis < 470){
    //Motor A backward
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    //Set Motor B backward
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    //Convert the declining Y-axis reading for going backward 
    //from 470 to 0 int 0 to 255 value for the PWM signal for
    //increasing the motor speed.
    motorSpeedA = map(yAxis, 470,0,0,255);
    motorSpeedB = map(yAxis, 470,0,0,255);
  } else if (yAxis > 550){
    //Set the motor forward
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    //Set Motor B forward
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    // Convert the increasing Y-axis readings for
    //going forward from 550 to 1023 into 0 to 255 value 
    //for the PWM signal for increasing the motor speed.
    motorSpeedA = map(yAxis, 550, 1023, 0, 255);
    motorSpeedB = map(yAxis, 550, 1023, 0, 255);
  } else {
    motorSpeedA = 0;
    motorSpeedB = 0;
  }
  //X-axis used for left and right control
  if (xAxis < 470){
    //Convert the declining X-axis readings from 470 to 0
    //into increasing 0 to 255 value.
    int xMapped = map(xAxis, 470,0,0, 255);
    // Move to left - decrease left motor speed, increase right motor speed
    motorSpeedA = motorSpeedA - xMapped;
    motorSpeedB = motorSpeedB + xMapped;
    motorSpeedA = constrain(motorSpeedA,0,255);
    motorSpeedB = constrain(motorSpeedB,0,255);
  }
  if (xAxis > 550){
    // Convert the increasing X-axis readings from 
    // 550 to 1023 into 0 to 255 value
    int xMapped = map(xAxis, 550, 1023,0,255);
    // Move right - decrease right motor speed, increase left motor speed
    motorSpeedA = motorSpeedA + xMapped;
    motorSpeedB = motorSpeedB - xMapped;
    // Confine the range from 0-255
    motorSpeedA = constrain(motorSpeedA,0,255);
    motorSpeedB = constrain(motorSpeedB,0,255);
  }
  /* To prevent buzzing. NOTE: Please test first the motors before
   * enabling this feature.
   * 
    */
  if (motorSpeedA < 70){
    motorSpeedA = 0;
  }
  if (motorSpeedB <70){
    motorSpeedB = 0;
  }
  
  analogWrite(enA, motorSpeedA);
  analogWrite(enB, motorSpeedB);
  
  buttonState1 = digitalRead(buttonPin1);
  buttonState2 = digitalRead(buttonPin2);
  //Rotate counterclockwise
  if (buttonState1 == HIGH){
    //Motor A backward
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    //Set Motor B forward
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    analogWrite(enA, 1000);
    analogWrite(enB, 1000);
    
  }
  if (buttonState2 == HIGH){
    //Motor A forward
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    //Set Motor B backward
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    analogWrite(enA, 1000);
    analogWrite(enB, 1000);
    
  }
}

void IRcontrolLeftRight(){
  if (sensorValue > 90){
    //Convert the declining X-axis readings from 470 to 0
    //into increasing 0 to 255 value.
    int xMapped = map(sensorValue, 90,180,0, 255);
    // Move to left - decrease left motor speed, increase right motor speed
    motorSpeedA = motorSpeedA - xMapped;
    delay(100);
    motorSpeedB = motorSpeedB + xMapped;
    delay(100);
    motorSpeedA = constrain(motorSpeedA,0,255);
    motorSpeedB = constrain(motorSpeedB,0,255);\
  }
  if (sensorValue < 70){
    // Convert the increasing X-axis readings from 
    // 550 to 1023 into 0 to 255 value
    int xMapped = map(sensorValue, 70, 0,0,255);
    // Move right - decrease right motor speed, increase left motor speed
    motorSpeedA = motorSpeedA + xMapped;
    delay(100);
    motorSpeedB = motorSpeedB - xMapped;
    delay(100);
    // Confine the range from 0-255
    motorSpeedA = constrain(motorSpeedA,0,255);
    motorSpeedB = constrain(motorSpeedB,0,255);
  }
  analogWrite(enA, motorSpeedA);
  analogWrite(enB, motorSpeedB);
}



