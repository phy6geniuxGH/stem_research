int redPin = 44;
int greenPin = 45;
int bluePin = 46;


void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
}

void loop() {
  setColor(255, 0, 0);
  delay(1000);
  setColor(0, 255, 0);
  delay(1000);
  setColor(0, 0, 255);
  delay(1000);
  setColor(255, 255, 255);
  delay(1000);
  setColor(170, 0, 255);
  delay(1000);
}

void setColor(int redValue, int greenValue, int BlueValue){
  analogWrite(redPin, redValue);
  analogWrite(greenPin, greenValue);
  analogWrite(bluePin, blueValue);
  
}









/*#include "pitches.h"

  const int speakerPin = 50;

  const int Pin[] = {42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 23};
  int notes[] = {NOTE_C4
               , NOTE_CS4
               , NOTE_D4
               , NOTE_DS4
               , NOTE_E4
               , NOTE_F4
               , NOTE_FS4
               , NOTE_G4
               , NOTE_GS4
               , NOTE_A4
               , NOTE_AS4
               , NOTE_B4
              };
  int PinState[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  void setup() {
  for (int i = 0; i < 12; i++) {
    pinMode(Pin[i], INPUT);
  }
  Serial.begin(9600);
  }

  void loop() {
  for (int i = 0; i < 12; i++) {
    PinState[i] = digitalRead(Pin[i]);

  }
  for (int j = 0; j < 12; j++) {
    if (PinState[j] == HIGH) {
      tone(speakerPin, notes[j], 50);
      Serial.println(notes[j]);
    }
  }
  }
*/
