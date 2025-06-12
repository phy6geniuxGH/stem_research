#include <Irscalib.h>

Irscalib irscalib;

void setup()
{
  irscalib.adjustSensor();
}

void loop()
{
  irscalib.calibratingSensor();
}

