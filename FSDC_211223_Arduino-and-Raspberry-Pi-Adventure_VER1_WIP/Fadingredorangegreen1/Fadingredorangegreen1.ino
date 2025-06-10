/*
  Fade

  This example shows how to fade an LED on pin 9 using the analogWrite()
  function.

  The analogWrite() function uses PWM, so if you want to change the pin you're
  using, be sure to use another PWM capable pin. On most Arduino, the PWM pins
  are identified with a "~" sign, like ~3, ~5, ~6, ~9, ~10 and ~11.

  This example code is in the public domain.

  http://www.arduino.cc/en/Tutorial/Fade
*/

int led5 = 5;
int led3 = 3;
int led7 = 9; // the PWM pin the LED is attached to
int brightness5 = 0;
int brightness3 = 0;
int brightness7 = 0;// how bright the LED is
int fadeAmount5 = 5;
int fadeAmount3 = 10;
int fadeAmount7 = 20;// how many points to fade the LED by

// the setup routine runs once when you press reset:
void setup() {
  // declare pin 9 to be an output:
  pinMode(led5, OUTPUT);
  pinMode(led3, OUTPUT);
  pinMode(led7, OUTPUT);
}

// the loop routine runs over and over again forever:
void loop() {
  // set the brightness of pin 9:
  analogWrite(led5, brightness5);

  // change the brightness for next time through the loop:
  brightness5= brightness5 + fadeAmount5;

  // reverse the direction of the fading at the ends of the fade:
  if (brightness5 <= 0 || brightness5 >= 255) {
    fadeAmount5 = -fadeAmount5;
  }
  // wait for 30 milliseconds to see the dimming effect
  delay(30);
  analogWrite(led3, brightness3);
  brightness3 = brightness3 + fadeAmount3;
  if (brightness3 <= 0 || brightness3 >= 255) {
    fadeAmount3 = -fadeAmount3;
  }
  delay(30);
  analogWrite(led7, brightness7);
   brightness7 = brightness7 + fadeAmount7;
  if (brightness7 <= 0 || brightness7 >= 255) {
    fadeAmount7 = -fadeAmount7;
  }
  delay(30);
}
