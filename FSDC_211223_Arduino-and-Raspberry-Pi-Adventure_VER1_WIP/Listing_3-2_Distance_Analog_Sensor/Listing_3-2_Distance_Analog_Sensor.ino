//Distance Alert

const int BLED = 9;
const int GLED = 10;
const int RLED = 11;
const int DIST = 0;

const int LOWER_BOUND = 250;
const int UPPER_BOUND = 500;

int val = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(BLED, OUTPUT);
  pinMode(GLED, OUTPUT);
  pinMode(RLED, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  val = analogRead(DIST);
  delay(10);
  //LED is Blue
  if (val > UPPER_BOUND){
    digitalWrite(RLED, LOW);
    digitalWrite(GLED, LOW);
    digitalWrite(BLED, HIGH);
  } else if (val < LOWER_BOUND){
    digitalWrite(RLED, HIGH);
    digitalWrite(GLED, LOW);
    digitalWrite(BLED, LOW);
  } else {
    digitalWrite(RLED, LOW);
    digitalWrite(GLED, HIGH);
    digitalWrite(BLED, LOW);
  }
  Serial.println(val);
  delay(500);
}
