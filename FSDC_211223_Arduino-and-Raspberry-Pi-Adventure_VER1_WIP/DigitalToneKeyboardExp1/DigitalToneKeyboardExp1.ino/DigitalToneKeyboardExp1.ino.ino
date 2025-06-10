#include "pitches.h"
//const int threshold = 10;
int notes[] = {
  NOTE_A4, NOTE_B4, NOTE_C3
};

void setup() {
  // put your setup code here, to run once:

}

void loop() {
  // put your main code here, to run repeatedly:
  //for(int thisbutton = 0; thisbutton < 3; thisSensor++){
    int buttonReading = digitalRead(thisbutton);
    if (buttonReading == HIGH){
      tone(8, notes[thisbutton], 20);
    }
  }

