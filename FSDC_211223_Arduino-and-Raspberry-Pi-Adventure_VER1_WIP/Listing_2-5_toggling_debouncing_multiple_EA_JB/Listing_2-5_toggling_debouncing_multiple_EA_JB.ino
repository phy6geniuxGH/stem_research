const int LED1         = 9;              // LED1 pin
const int LED2         = 10;             // LED2 pin
const int BUTTON1      = 2;              // Button pin
const int BUTTON2      = 4;
boolean lastButton1    = LOW;            // Button's last state
boolean currentButton1 = LOW;            // Button's current state
boolean lastButton2    = LOW;            // Button's last state
boolean currentButton2 = LOW;            // Button's current state

boolean ledOn1         = false;          // LED's present state
boolean ledOn2         = false;          // LED's present state

void setup() {
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(BUTTON1, INPUT);
  pinMode(BUTTON2, INPUT);
}

/*
 * Debouncing function
 */
boolean debounce(boolean last, int button){
  boolean current = digitalRead(button);  // Read Button's state
  if (last != current){                   // If note the same,
    delay(5);                             // Wait for 5 milliseconds
    current = digitalRead(button);        // Read the button's state again
  }
  return current;                         // Assign the new reading to current variable and then return.
}

void loop() {
  currentButton1 = debounce(lastButton1, BUTTON1);
  currentButton2 = debounce(lastButton2, BUTTON2);
  
  if(lastButton1 == LOW && currentButton1 == HIGH){
    ledOn1 = !ledOn1;
  }
  lastButton1 = currentButton1;
  
  if(lastButton2 == LOW && currentButton2 == HIGH){
    ledOn2 = !ledOn2;
  }
  lastButton2 = currentButton2;

  digitalWrite(LED1, ledOn1);
  digitalWrite(LED2, ledOn2);
}
