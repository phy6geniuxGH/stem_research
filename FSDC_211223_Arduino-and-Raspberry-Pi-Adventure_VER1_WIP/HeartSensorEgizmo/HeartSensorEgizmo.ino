int heartsensor = A0;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(38400);
  pinMode(heartsensor, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()>0){
    Serial.read(heartsensor);
  }
  //Serial.println();
  delay(5);
}
