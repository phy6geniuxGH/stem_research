//Two photresistors

int sensePin0 = 0;
int sensePin1 = 1;
int ledPin9 = 9;
int ledPin10 = 10;

void setup() {
  analogReference(DEFAULT);
  pinMode(ledPin9, OUTPUT);
  pinMode(ledPin10, OUTPUT);
  //Serial.begin(9600);
}

void loop() {
  int val1 = analogRead(sensePin0);
  int val2 = analogRead(sensePin1);
  //Serial.println(val2);
  //delay(500);
  if(val1 <= 800){
    digitalWrite(ledPin9, HIGH);
  } else {
    digitalWrite(ledPin9, LOW);
  }
  if(val2 >= 300){
    digitalWrite(ledPin10, HIGH);
  } else {
    digitalWrite(ledPin10, LOW);
  }
}
