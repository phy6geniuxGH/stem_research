//Arduino Two Motor Control with Potentiometer and a Button

#define enA 9
#define enB 10
#define in1 6
#define in2 7
#define in3 12
#define in4 11
#define button 4

int rotDirection = 0;
int pressed = false;

void setup(){
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(button, OUTPUT);
  //Set initial rotation direction (changing these will result to a different direction rotation
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}
void loop(){
  int potValue = analogRead(A0); //Read potentiometer value
  int pwmOutput = map(potValue, 0, 1023, 0, 255); //map the potentiometer
                                                  //value from 0 to 255
  analogWrite(enA, pwmOutput); //Send PWM signal to L298N Enable pin
  analogWrite(enB, pwmOutput);
  //Read button - Debounce
  if (digitalRead(button) == true){
    pressed = !pressed;
  }
  while (digitalRead(button) == true);
  delay(20);

  if(pressed == true & rotDirection ==0){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    rotDirection = 1;
    delay(20);
  }
  //If button is pressed - changed rotation direction
  if (pressed == false & rotDirection == 1){
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    rotDirection = 0;
    delay(20);
  }
}


