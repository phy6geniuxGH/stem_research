/*
 * Francis's First Arduino Program
 * 
 */

int ledPin = 13;

void setup() {
  // initialize pin outputs
  pinMode(ledPin, OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(ledPin, HIGH);
  delay(1000);
  digitalWrite(ledPin, LOW);
  delay(1000);
  
}
