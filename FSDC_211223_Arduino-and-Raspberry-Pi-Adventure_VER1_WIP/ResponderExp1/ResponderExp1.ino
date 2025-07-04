/*
  DigitalReadSerial

  Reads a digital input on pin 2, prints the result to the Serial Monitor

  This example code is in the public domain.

  http://www.arduino.cc/en/Tutorial/DigitalReadSerial
*/

// digital pin 2 has a pushbutton attached to it. Give it a name:
int pushButton1 = 5;
int pushButton2 = 4;
int pushButton3 = 3;
int pushReset = 2;
int led1 = 8;
int led2 = 7;
int led3 = 6;
int button1 = HIGH;
int button2 = LOW;
int brightness = 0;
int fade = 5;
//int myPins[] = {6,7,8};

// the setup routine runs once when you press reset:
void setup() {
  // make the pushbutton's pin an input:
  pinMode(pushButton1, INPUT);
  pinMode(pushButton2, INPUT);
  pinMode(pushButton3, INPUT);
  pinMode(pushReset, INPUT);
  pinMode(led1, OUTPUT);
  pinMode(led2, OUTPUT);
  pinMode(led3, OUTPUT);
  //pinMode(myPins, OUTPUT);
  
}

void loop() {
  int buttonState1 = digitalRead(pushButton1);
  if (buttonState1 == button1){ //push the button, it will remain on.
    digitalWrite(led1, HIGH);
  }
  
  int buttonState2 = digitalRead(pushButton2);
  digitalWrite(led2, buttonState2);
  
  int buttonState3 = digitalRead(pushButton3);
  if (buttonState3 == button1){
    analogWrite(led3, brightness);
    brightness = brightness + fade;
    if (brightness <= 0 || brightness >= 255){
      fade = -fade;
    } else {
      buttonState3 = button1; // with this only, the button will fade when holding the button and stays with its current brightness once released.
      while (buttonState3 /= button1){ //with this, the led3 will do fade looping.
        analogWrite(led3, brightness);
        brightness = brightness + fade;
        if (brightness <= 0 || brightness >= 255){
          fade = -fade;
          break; //putting a break here will make the led3 min -> max brightness by one button push, and vice versa.
        }
        delay(20);
      }
    }
    delay(20);
  }
  //digitalWrite(led3, buttonState3);
  int buttonState4 = digitalRead(pushReset); //this is our of the void loop. It acts as a reset.
  if (buttonState4 == button1){ //push the button, led1 will turn off
    digitalWrite(led1, LOW);
    analogWrite(led3, LOW);
  }
  //
  // digitalWrite(led1, buttonState4);   // enabling these will result to a dimmer led lights. Reason: Connected to 2 10k-ohm pull-down resistors of two buttons, instead of 1
  // digitalWrite(led2, buttonState4);
  // (led3, buttonState4);
  // print out the state of the button:
  delay(1);        // delay in between reads for stability
}
