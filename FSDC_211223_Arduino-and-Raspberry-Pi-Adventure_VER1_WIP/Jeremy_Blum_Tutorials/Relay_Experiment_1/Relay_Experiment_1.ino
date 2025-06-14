int in1 = 7;
int in2 = 8;

void setup() {
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  digitalWrite(in1, HIGH);
  digitalWrite(in2, HIGH);
  //this will be read as a logic low, since the module works inversely
}
void loop() {

  digitalWrite(in1, LOW);
  delay(1000);
  digitalWrite(in2, LOW);
  delay (1000);
  digitalWrite(in1, HIGH);
  delay(1000);
  digitalWrite(in2, HIGH);
  delay(1000);
}
