int redPin = 23;
int greenPin = 22;
int bluePin = 21;
int ledPin = 13;
int rate = 1;
int scalar = 1;
void setup() {

  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  rate = rate + scalar * 5;
  if (rate < 0 || rate > 255) {
    scalar = -scalar;
  }
  /*
  digitalWrite(redPin, HIGH);
  delay(rate);
  digitalWrite(greenPin, HIGH);
  delay(rate);
  digitalWrite(bluePin, HIGH);
  delay(rate);
  digitalWrite(ledPin, HIGH);
  delay(rate);
  digitalWrite(redPin, LOW);
  delay(rate);
  digitalWrite(greenPin, LOW);
  delay(rate);
  digitalWrite(bluePin, LOW);
  delay(rate);
  digitalWrite(ledPin, LOW);
  delay(rate);*/
  analogWrite(redPin, rate);
  analogWrite(greenPin, rate);
  analogWrite(bluePin,rate);
  delay(10);
}
