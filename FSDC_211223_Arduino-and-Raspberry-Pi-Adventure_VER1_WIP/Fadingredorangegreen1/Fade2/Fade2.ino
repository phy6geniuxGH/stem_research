int led5 = 5;
int led7 = 3;
int led9 = 9;
int brightness = 0;
int fadeAmount = 5;

void setup() {
  pinMode(led5, OUTPUT);
}
void loop(){
  analogWrite(led5, brightness);
  analogWrite(led7, brightness);
  analogWrite(led9, brightness);
  brightness = brightness + fadeAmount;
  if (brightness <= 0 || brightness >=255){
    fadeAmount = - fadeAmount;
    }
  delay(20);
}

