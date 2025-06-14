//switching on anf off with bouncing

int switchPin = 8;
int ledPin = 13;
// two codes that will keep track of the button states
boolean lastButton = LOW;
boolean ledOn = false;

void setup() {
  pinMode(switchPin, INPUT);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  //we wanna see if switchPin is HIGH AND lastButton is LOW
  //Keeping track of the states
  if(digitalRead(switchPin) == HIGH && lastButton == LOW){
    ledOn = !ledOn; //reverse the state of ledOn. 
    lastButton = HIGH; //assign the new state of the lastButton
  } else {
    lastButton = digitalRead(switchPin);
  }
  digitalWrite(ledPin, ledOn); //will turn on the LED if the switchPin input is HIGH.
}
