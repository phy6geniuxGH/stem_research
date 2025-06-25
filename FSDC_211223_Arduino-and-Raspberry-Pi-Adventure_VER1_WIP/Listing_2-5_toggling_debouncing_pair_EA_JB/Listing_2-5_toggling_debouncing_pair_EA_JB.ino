const int LED1        = 9;              // LED1 pin
const int LED2        = 10;             // LED2 pin
const int BUTTON      = 2;              // Button pin
boolean lastButton    = LOW;            // Button's last state
boolean currentButton = LOW;            // Button's current state
boolean ledOn         = false;          // LED's present state

void setup() {
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(BUTTON, INPUT);
}

/*
 * Debouncing function
 */
boolean debounce(boolean last){
  boolean current = digitalRead(BUTTON);  // Read Button's state
  if (last != current){                   // If note the same,
    delay(5);                             // Wait for 5 milliseconds
    current = digitalRead(BUTTON);        // Read the button's state again
  }
  return current;                         // Assign the new reading to current variable and then return.
}

void loop() {
  currentButton = debounce(lastButton);
  if(lastButton == LOW && currentButton == HIGH){
    ledOn = !ledOn;
  }
  lastButton = currentButton;

  digitalWrite(LED1, ledOn);
  digitalWrite(LED2, !ledOn);
}
