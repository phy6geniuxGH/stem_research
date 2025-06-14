int switchPin = 2;
int ledPin = 13;

void setup() {
  // put your setup code here, to run once:
  pinMode(switchPin, INPUT);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(digitalRead(switchPin) == HIGH){
    digitalWrite(ledPin, HIGH);
    delay(50);
  } else {
    digitalWrite(ledPin, LOW);
    delay(50);
  }
}
