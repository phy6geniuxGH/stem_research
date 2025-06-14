//Adaptive Brightness

int sensePin0 = 0;
int sensePin1 = 1;
int ledPin9 = 9;


void setup() {
  analogReference(DEFAULT);
  pinMode(ledPin9, OUTPUT);
  Serial.begin(9600);
  
}

void loop() {
  int val1 = analogRead(sensePin1);
  Serial.println(val1);
  delay(500);
  int val = analogRead(sensePin0);
  //int val2 = analogRead(sensePin1);
  val = constrain(val,150,850); //for values <750 or >900, it resets them back to 750 and 900.
  int ledLevel = map(val, 150, 850, 255, 0); //maps 750 to 255 and 900 to 0.
  analogWrite(ledPin9, ledLevel);
 
}
