//switching through debouncing
int switchPin = 8;
int ledPin = 13;
// two codes that will keep track of the button states
boolean lastButton = LOW;
boolean ledOn = false;
// put a currentButton variable:
boolean currentButton = LOW;

void setup() {
  pinMode(switchPin, INPUT);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

//add a function for debouncing - the class type is boolean.
// Function: Reads the button state, compare it to the last button state, 
boolean debounce(boolean last){ // the argument of the function takes logic type variable of any name.
  boolean current = digitalRead(switchPin);
  // add and argument to compare button states, to tell if someone hit the switch:
  if (last != current){
    delay(50); // add a delay, which is still faster than the human reflex (5 ms).
    current = digitalRead(switchPin);
  }
  return current; //returns the value of the current state
}

void loop() {
  currentButton = debounce(lastButton); //use the defined function "debounce"
                                        //to check the current button state.
  Serial.println(currentButton);
  if(lastButton == LOW && currentButton == HIGH){
    ledOn = !ledOn; //reverse the state of ledOn. 
    //lastButton = HIGH; //not needed now, due to the debounce function takes care of it.
  }
    lastButton = currentButton;
  digitalWrite(ledPin, ledOn); //will turn on the LED if the switchPin input is HIGH.
}
