int LED[]                = {9, 10};        // LED pins
int BUTTON[]             = {2, 4};         // Button pin
boolean lastButton[]     = {LOW, LOW};     // Button's last state
boolean currentButton[]  = {LOW, LOW};     // Button's current state
boolean ledOn[]          = {false, false}; // LED's present state

void setup() {
  for (int i = 0; i < sizeof(LED)/sizeof(int); i++){
    pinMode(LED[i], OUTPUT);
    pinMode(BUTTON[i], INPUT);
  }
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
  for (int i = 0; i < sizeof(LED)/sizeof(int); i++){
    currentButton[i] = debounce(lastButton[i], BUTTON[i]);
    
    if(lastButton[i] == LOW && currentButton[i] == HIGH){
      ledOn[i] = !ledOn[i];
    }
    lastButton[i] = currentButton[i];
  
    digitalWrite(LED[i], ledOn[i]);
    
  }
}
