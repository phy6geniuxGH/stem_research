#include "pitches.h"

int delayTime = 50;
int colorPin[] = {44, 45, 46, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

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
  for (int i = 0; i < 15; i++) {
    pinMode(colorPin[i], OUTPUT);
  }

  for (int i = 0; i < 12; i++) {
    pinMode(Pin[i], INPUT);
  }
  Serial.begin(9600);
}


void loop() {


  setColor(255, 0, 0);
  delay(delayTime);
  setColor(0, 255, 0);
  delay(delayTime);
  setColor(0, 0, 255);
  delay(delayTime);
  setColor(255, 255, 255);
  delay(delayTime);
  setColor(170, 0, 255);
  delay(delayTime);

}

void setColor(int redValue, int greenValue, int blueValue) {
  analogWrite(colorPin[0], redValue);
  analogWrite(colorPin[1], greenValue);
  analogWrite(colorPin[2], blueValue);
  analogWrite(colorPin[3], redValue);
  analogWrite(colorPin[4], greenValue);
  analogWrite(colorPin[5], blueValue);
  analogWrite(colorPin[6], redValue);
  analogWrite(colorPin[7], greenValue);
  analogWrite(colorPin[8], blueValue);
  analogWrite(colorPin[9], redValue);
  analogWrite(colorPin[10], greenValue);
  analogWrite(colorPin[11], blueValue);
  analogWrite(colorPin[12], redValue);
  analogWrite(colorPin[13], greenValue);
  analogWrite(colorPin[14], blueValue);


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
