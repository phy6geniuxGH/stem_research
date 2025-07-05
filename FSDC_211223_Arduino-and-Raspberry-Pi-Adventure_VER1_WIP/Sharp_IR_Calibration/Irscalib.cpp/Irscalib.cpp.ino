/*
 * Practice for Library Making
 */


#include "Arduino.h"
#include "Irscalib.h"

Irscalib::Irscalib(int sensorPin)
{
  _sensorPin = sensorPin;
}
Irscalib::Irscalib(int sensorValue)
{
  _sensorValue = sensorValue;
}
Irscalib::Irscalib(int sensorMin)
{
  _sensorMin = sensorMin;
}
Irscalib::Irscalib(int sensorMax)
{
  _sensorMax = sensorMax;
}

void Irscalib::adjustSensor()
{
  Serial.begin(9600);
  while (millis() < 10000){
    sensorValue = analogRead(_sensorPin);
    if(_sensorValue > _sensorMax){
      _sensorMax = _sensorValue;
    }
    if(_sensorValue < _sensorMin){
      _sensorMin = _sensorValue;
    }
    Serial.println(_sensorMin);
    Serial.println(_sensorMax);
    delay(10);
}

void Irscalib::calibratingSensor()
{
  _sensorValue = analogRead(_sensorPin);
  _sensorValue = map(_sensorValue, _sensorMin, _sensorMax, 180, 0); 
  _sensorValue = constrain(_sensorValue, 0, 255);
  Serial.println(_sensorValue);
  delay(100);
}

