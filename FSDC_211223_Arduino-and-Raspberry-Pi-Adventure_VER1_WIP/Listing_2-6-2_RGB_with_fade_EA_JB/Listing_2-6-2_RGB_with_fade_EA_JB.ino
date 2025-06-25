int LED[]                = {9, 10, 11};        // LED pins
int BUTTON[]             = {2, 4};         // Button pin
boolean lastButton[]     = {LOW, LOW};     // Button's last state
boolean currentButton[]  = {LOW, LOW};     // Button's current state
int ledMode[]            = {0}; // LED's present state

void setup() {
  for (int i = 0; i < sizeof(LED)/sizeof(int); i++){
    pinMode(LED[i], OUTPUT);
  }
  for (int i = 0; i < sizeof(LED)/sizeof(int); i++){
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


/*
 * LED Mode Selection
 */
void setMode(int mode){
  //RED
  if(mode == 1){
    digitalWrite(LED[0], HIGH);
    digitalWrite(LED[1], LOW);
    digitalWrite(LED[2], LOW);
  }

  //GREEN
  else if(mode == 2){
    digitalWrite(LED[0], LOW);
    digitalWrite(LED[1], HIGH);
    digitalWrite(LED[2], LOW);
  }

  //BLUE
  else if(mode == 3){
    digitalWrite(LED[0], LOW);
    digitalWrite(LED[1], LOW);
    digitalWrite(LED[2], HIGH);
  }
  
  //PURPLE (RED + BLUE)
  else if(mode == 4){
    analogWrite(LED[0], 127);
    analogWrite(LED[1], 0);
    analogWrite(LED[2], 127);
  }

  //TEAL (BLUE + GREEN)
  else if(mode == 5){
    analogWrite(LED[0], 0);
    analogWrite(LED[1], 127);
    analogWrite(LED[2], 127);
  }

  //ORANGE (GREEN + RED)
  else if(mode == 6){
    analogWrite(LED[0], 255);
    analogWrite(LED[1], 10);
    analogWrite(LED[2], 0);
  }

  //WHITE (GREEN + RED + BLUE)
  else if(mode == 7){
    analogWrite(LED[0], 255);
    analogWrite(LED[1], 100);
    analogWrite(LED[2], 100);
  }

  //OFF Mode (mode == 0)
  else {
    digitalWrite(LED[0], LOW);
    digitalWrite(LED[1], LOW);
    digitalWrite(LED[2], LOW);
  }
}


void loop() {
  currentButton[0] = debounce(lastButton[0], BUTTON[0]);
  if (lastButton[0] == LOW && currentButton[0] == HIGH){
    ledMode[0]++;
  }
  lastButton[0] = currentButton[0];

  //resetting the counter to 0
  if (ledMode[0] == 8) ledMode[0] = 0;
  setMode(ledMode[0]);

  currentButton[1] = debounce(lastButton[1], BUTTON[1]);
  if (lastButton[1] == LOW && currentButton[1] == HIGH){
    for (int i = 100; i<=1000; i+=100){
      digitalWrite(LED[0], HIGH);
      delay(i);
      digitalWrite(LED[0], LOW);
      delay(i);
    }
    for (int i = 0; i<256; i++){
      analogWrite(LED[0], i);
      delay(10);
    }
  }
  lastButton[1] = currentButton[1];

  
  
}
