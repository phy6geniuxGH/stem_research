//Motor and PWM
int motorPin = 10;

void setup() {
  pinMode(motorPin, OUTPUT);

}

void loop() {
  //Accelerate Motor from 0 ->255
  //new programming tactic: for loop
  for(int i=0; i<=255;i++){ //as long i <= 255, the 
                             //value of i from 0 will increase by 1.
    analogWrite(motorPin, i);
    delay(10);
  }

  //Hold at Top Speed
  delay(10000);

  //Decrease Speed from 255 -> 0
  for(int i=255; i>=0; i--){ //same tactic in reverse
    analogWrite(motorPin, i);
    delay(10);
  }
  //Hold at Zero
  delay(500);
}
