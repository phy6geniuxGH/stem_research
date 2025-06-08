
int led_3 = 3;
int led_5 = 5;
int led_7 = 7;

// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin led_BUILTIN as an output.
  pinMode(led_3, OUTPUT);
  pinMode(led_5, OUTPUT);
  pinMode(led_7, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {
  digitalWrite(led_3, HIGH);   // turn the led on (HIGH is the voltage level)
  delay(50);                       // wait for a second
  digitalWrite(led_3, LOW);    // turn the led off by making the voltage LOW
  delay(50);                       // wait for a second
  digitalWrite(led_5, HIGH);   // turn the led on (HIGH is the voltage level)
  delay(50);                       // wait for a second
  digitalWrite(led_5, LOW);    // turn the led off by making the voltage LOW
  delay(50); 
  digitalWrite(led_7, HIGH);   // turn the led on (HIGH is the voltage level)
  delay(50);                       // wait for a second
  digitalWrite(led_7, LOW);    // turn the led off by making the voltage LOW
  delay(50); 
  digitalWrite(led_5, HIGH);   // turn the led on (HIGH is the voltage level)
  delay(50);                       // wait for a second
  digitalWrite(led_5, LOW);    // turn the led off by making the voltage LOW
  delay(50); 
}
