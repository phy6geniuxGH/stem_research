//This works using TC74 Temp sensor

//include the wire library
#include <Wire.h>
//set the address of the temp sensor
int temp_address = 0x18;

void setup(){
  Serial.begin(9600);
  Wire.begin();
}
void loop (){
  //Send a request
  //Start talking
  Wire.beginTransmission(temp_address);
  //Ask for Register zero
  Wire.write(0); //send renamed to write
  //Complete Transmission
  Wire.endTransmission();
  //Request 1 byte
  Wire.requestFrom(temp_address, 1);
  //wait for response
  while(Wire.available() == 0);
  //get the temp
  int c = Wire.read(); //receive renamed to read
  //convert from celcius to farenheit
  int f = round(c*9.0/5.0 + 32.0);
  //print the results
  Serial.print(c);
  Serial.print("C,");
  Serial.print(f);
  Serial.println("F.");
  //delay, then do it again
  delay(500);
}

