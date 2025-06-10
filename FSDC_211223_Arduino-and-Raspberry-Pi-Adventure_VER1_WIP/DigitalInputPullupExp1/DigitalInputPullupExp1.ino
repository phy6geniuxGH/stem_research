const int pin2 = 2;
const int pin13 = 13;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(pin2, INPUT_PULLUP);
  pinMode(pin13, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  int sensorVal = digitalRead(pin2);
  Serial.println(sensorVal);
  //pull-up means the pushbutton's logic is inverted.
  if (sensorVal == HIGH){
    digitalWrite(pin13, HIGH);
  } else {
    digitalWrite(pin13, LOW);
  }
}
