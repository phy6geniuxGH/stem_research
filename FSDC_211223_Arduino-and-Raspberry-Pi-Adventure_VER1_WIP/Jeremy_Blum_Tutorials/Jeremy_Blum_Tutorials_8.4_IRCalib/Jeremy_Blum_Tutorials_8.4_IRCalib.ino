//IR Calibration

const int sensorPin = A0;

int sensorValue = 0;
int sensorMin = 6;
int sensorMax = 506;

void setup(){
  adjustSensor();
}
void loop(){
  calibratingSensor();
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

