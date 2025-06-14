void setup() {
  Serial.begin(9600);
}

void loop() {
  // have the arduino wait to receive input
  while(Serial.available() == 0);
  //Read the Input
  int val = Serial.read() - '0';
  //Reason: 0 is 48 in ASCII. 1 is 49. 49-48 = 1, and then returns 1. 
  //'0' means that it is character(char).  
  //Echo the input
  Serial.println(val);
  //This code doesn't send correct input - go to data types.
  //ASCII
}
