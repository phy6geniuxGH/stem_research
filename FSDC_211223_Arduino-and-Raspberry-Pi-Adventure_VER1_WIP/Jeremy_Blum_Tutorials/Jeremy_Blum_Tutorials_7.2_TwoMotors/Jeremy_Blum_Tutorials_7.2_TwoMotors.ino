//Motor and PWM
int motorPin9 = 9;
int motorPin10 = 10;

void setup() {
  pinMode(motorPin9, OUTPUT);
  pinMode(motorPin10, OUTPUT);

}

void loop() {
  //Accelerate Motor from 0 ->255
  //new programming tactic: for loop
  for(int i=0; i<=255;i++){ //as long i <= 255, the 
                             //value of i from 0 will increase by 1.
    analogWrite(motorPin9, i);
    analogWrite(motorPin10, i);
    delay(10);
  }

  //Hold at Top Speed
  delay(1000);

  //Decrease Speed from 255 -> 0
  for(int i=255; i>=0; i--){ //same tactic in reverse
    analogWrite(motorPin9, i);
    analogWrite(motorPin10, i);
    delay(10);
  }
  //Hold at Zero
  delay(500);
}
