#include <IRremote.h>
int RECV_PIN = 9;
int relayOut = 7;
int buttonState ;
IRrecv irrecv(RECV_PIN);

decode_results results;

int redPin = 5;
int greenPin = 6;

void setup() {
  Serial.begin(9600);
  irrecv.enableIRIn();
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(relayOut, OUTPUT);
  buttonState = HIGH;
}

void loop() {
  buttonState = digitalRead(relayOut);
  if (irrecv.decode(&results)) {
    if (results.value == 0x4ED5545A) {
      setColor(0, 255);
      digitalWrite(relayOut, LOW); // Activates the relay
      delay(100);
    }
    if (results.value == 0x2B77163A) {
      digitalWrite(relayOut, HIGH); // Deactivates the relay
      setColor(255, 0);
      delay(100);
    }
    if (results.value == 0x9774845A) {
      setColor(255, 200);
      delay(100);
    }
    irrecv.resume();
  }
  delay(100);

}
void setColor(int red, int green) {
  analogWrite(redPin, red);
  analogWrite(greenPin, green);
}
