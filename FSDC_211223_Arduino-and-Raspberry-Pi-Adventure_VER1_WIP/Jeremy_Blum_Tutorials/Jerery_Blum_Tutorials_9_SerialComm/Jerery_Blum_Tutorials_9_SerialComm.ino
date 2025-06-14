void setup() {
  Serial.begin(9600);
}

void loop() {
  // have the arduino wait to receive input
  while(Serial.available() == 0);
  //Read the Input
  int val = Serial.read();
  //Echo the input
  Serial.println(val);
  //This code doesn't send correct input - go to data types.
  //ASCII
}
