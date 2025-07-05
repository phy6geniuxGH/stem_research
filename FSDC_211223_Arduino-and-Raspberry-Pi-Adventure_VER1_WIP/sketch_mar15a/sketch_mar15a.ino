// led is the givenname for digital pin 13.
int led = 13;

// the setting up of pins.
void setup() {                
  // initialize the digital pin as an output.
  pinMode(led, OUTPUT);     
}

// the loop is where your program runs repeatedly.
void loop() {
  digitalWrite(led, HIGH);   // turn the LED ON (HIGH)
  delay(1000);               // wait for a second
  digitalWrite(led, LOW);    // turn the LED OFF (LOW)
  delay(1000);               // wait for a second
}
