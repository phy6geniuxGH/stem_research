#include "Stepper.h"
#include<LiquidCrystal.h>
LiquidCrystal lcd(31,33,35,37,39,41);
#define STEPS  32   // Number of steps for one revolution of Internal shaft
                    // 2048 steps for one revolution of External shaft

volatile boolean TurnDetected;  // need volatile for Interrupts
volatile boolean rotationdirection;  // CW or CCW rotation

int s=10;
int i;
const int PinCLK=2;   // Generating interrupts using CLK signal
const int PinDT=3;    // Reading DT signal
const int PinSW=4;    // Reading Push Button switch

int RotaryPosition=0;    // To store Stepper Motor Position

int PrevPosition;     // Previous Rotary position Value to check accuracy
int StepsToTake;      // How much to move Stepper

// Setup of proper sequencing for Motor Driver Pins
// In1, In2, In3, In4 in the sequence 1-3-2-4
Stepper small_stepper(STEPS, 8, 10, 9, 11);

// Interrupt routine runs if CLK goes from HIGH to LOW
void isr ()  {
  delay(2);  // delay for Debouncing
  if (digitalRead(PinCLK))
    rotationdirection= digitalRead(PinDT);
  else
    rotationdirection= !digitalRead(PinDT);
  TurnDetected = true;
}

void setup ()  {
  
pinMode(PinCLK,INPUT);
pinMode(PinDT,INPUT);  
pinMode(PinSW,INPUT);
pinMode(6,OUTPUT);
digitalWrite(PinSW, HIGH); // Pull-Up resistor for switch
attachInterrupt (0,isr,FALLING); // interrupt 0 always connected to pin 2 on Arduino UNO
lcd.begin(16,2);
digitalWrite(6,LOW);
for(i=5;i>=1;i-=1){
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Starting in");
  lcd.setCursor(0,1);
  lcd.print(i);
  delay(1000);
}
lcd.clear();
lcd.setCursor(1,0);
lcd.print("Ready to use!");
digitalWrite(6,HIGH);
}

void loop ()  {
  small_stepper.setSpeed(900); //Max seems to be 700
  if (!(digitalRead(PinSW))) {   // check if button is pressed
    if (RotaryPosition == 0) {  // check if button was already pressed
    } else {
        digitalWrite(6,LOW);
        lcd.clear();
        lcd.setCursor(0,0);
        lcd.print("Position reset");
        lcd.setCursor(0,1);
        lcd.print("in process");
        small_stepper.step(-(RotaryPosition*s));
        lcd.clear();
        lcd.setCursor(0,0);
        lcd.print("Position reset");
        lcd.setCursor(0,1);
        lcd.print("done!");
        delay(1500);
        digitalWrite(6,HIGH);
        lcd.clear();
        lcd.setCursor(0,0);
        lcd.print("Ready to use");
        lcd.setCursor(0,5);
        lcd.print("again!");
        RotaryPosition=0; // Reset position to ZERO
      }
    }

  // Runs if rotation was detected
  if (TurnDetected)  {
    PrevPosition = RotaryPosition; // Save previous position in variable
    if (rotationdirection) {
      RotaryPosition=RotaryPosition-1;} // decrase Position by 1
    else {
      RotaryPosition=RotaryPosition+1;} // increase Position by 1

    TurnDetected = false;  // do NOT repeat IF loop until new rotation detected

    // Which direction to move Stepper motor
    if ((PrevPosition + 1) == RotaryPosition) { // Move motor CW
      StepsToTake=s; 
      small_stepper.step(StepsToTake);
              lcd.clear();
        lcd.setCursor(0,0);
        lcd.print("Steps taken");
        lcd.setCursor(0,1);
        lcd.print(RotaryPosition*s);
    }

    if ((RotaryPosition + 1) == PrevPosition) { // Move motor CCW
      StepsToTake=-s;
      small_stepper.step(StepsToTake);
              lcd.clear();
        lcd.setCursor(0,0);
        lcd.print("Steps taken");
        lcd.setCursor(0,1);
        lcd.print(RotaryPosition*s);
    }
  }
}

