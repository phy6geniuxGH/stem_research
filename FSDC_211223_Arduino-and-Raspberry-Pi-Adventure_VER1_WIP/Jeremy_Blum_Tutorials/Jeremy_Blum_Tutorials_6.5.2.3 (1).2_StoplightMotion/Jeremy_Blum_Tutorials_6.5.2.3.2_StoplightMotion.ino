//Infrared Sensor + Photodiode
//motion and light sensing

//Define Pins
int motionPin = 0;
int lightPin = 1;
int ledPin9 = 9;
int ledPin10 = 10;
int ledPin11 = 11;

//Distance Variables
int lastDist = 0;
int currentDist = 0;

//State Variables
int redtogreen = 1;
int greentored = 0;

//Threshold for Movement
int thresh = 100;

void setup(){
  pinMode(ledPin9, OUTPUT);
  pinMode(ledPin10, OUTPUT);
  pinMode(ledPin11, OUTPUT);
}

void loop(){
  int lightVal = analogRead(lightPin);
  currentDist = analogRead(motionPin);
  //Does the current distance deviate from the last distance by more than the threshold?
  if ((currentDist > lastDist + thresh || currentDist < lastDist - thresh) && lightVal < 800){
    if (redtogreen == 1){
      greentored = 1;
     }
    digitalWrite(ledPin10, LOW);
    delay(10);
    if (redtogreen == 1){
      statefunction1();
      redtogreen = 0;
    }
    digitalWrite(ledPin9, HIGH);
    delay(1000);
    
  } else {
    if (greentored == 0){
      redtogreen = 0;
     }
    digitalWrite(ledPin9, LOW);
    delay(10);
    if (greentored == 0){
      statefunction2();
      greentored = 1;
    }
    digitalWrite(ledPin10, HIGH);
    delay(1000);
  }
  lastDist = currentDist;
}

void statefunction1(){
  digitalWrite(ledPin11, HIGH);
  delay(500);
  digitalWrite(ledPin11, LOW);
  delay(100);
  greentored = 0;
}

void statefunction2(){
  digitalWrite(ledPin11, HIGH);
  delay(500);
  digitalWrite(ledPin11, LOW);
  delay(100);
  redtogreen = 1;
}

