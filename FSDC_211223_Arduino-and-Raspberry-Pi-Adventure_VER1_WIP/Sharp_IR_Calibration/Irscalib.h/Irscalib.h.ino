/*
Practice Library Making (July 6, 2018)
*/
#ifndef Irscalib.h
#define Irscalib.h

#include "Arduino.h"
class Irscalib
{
  public:
    Irscalib(int sensorPin);
    Irscalib(int sensorValue);
    Irscalib(int sensorMin);
    Irscalib(int sensorMax);
    void adjustSensor();
    void calibratingSensor();
  private:
    int _sensorPin;
    int _sensorValue;
    int _sensorMin;
    int _sensorMax;
};

#endif
