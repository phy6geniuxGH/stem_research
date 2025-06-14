//Sharp IR Lib

/* where:
   ir: the pin where your sensor is attached.
    25: the number of readings the library will make before calculating an average distance.
    93: the difference between two consecutive measurements to be taken as valid (in %)
    model: is an int that determines your sensor:  1080 for GP2Y0A21Y, 20150 for GP2Y0A02
     */
#include <SharpIR.h>

int ir = 0;
SharpIR sharp(ir, 20150);

void setup(){
  Serial.begin(9600);
}

void loop(){
  int dist = sharp.distance();
  int pos = map(dist, 0, 1023, 15, 150);
  pos = constrain(pos, 0, 255);
  Serial.println(pos);
  delay(100);
}

