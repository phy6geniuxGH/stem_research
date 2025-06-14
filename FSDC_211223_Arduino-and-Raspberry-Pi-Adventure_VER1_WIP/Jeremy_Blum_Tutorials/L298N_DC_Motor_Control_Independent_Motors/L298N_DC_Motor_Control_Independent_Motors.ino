//Arduino Two Motor Control with Potentiometer and a Button

#define enA 8
#define enB 13
#define in1 9
#define in2 10
#define in3 11
#define in4 12
#define button1 4
#define button2 2

int rotDirection1 = 0;
int rotDirection2 = 0;
int pressed1 = false;
int pressed2 = false;

void setup(){
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(button1, OUTPUT);
  pinMode(button2, OUTPUT);
  //Set initial rotation direction (changing these will result to a different direction rotation
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}
void loop(){
  int potValue1 = analogRead(A0); //Read potentiometer 1 value
  int potValue2 = analogRead(A1); //Read potentiometer 2 value
  int pwmOutput1 = map(potValue1, 0, 1023, 0, 255);
  int pwmOutput2 = map(potValue2, 0, 1023, 0, 255); 
  analogWrite(enA, pwmOutput1); //Send PWM signal to L298N Enable pin
  analogWrite(enB, pwmOutput2);
  //Read button - Debounce
  if (digitalRead(button1) == true){
    pressed1 = !pressed1;
  }
  while (digitalRead(button1) == true);
  delay(20);

  if (digitalRead(button2) == true){
    pressed2 = !pressed2;
  }
  while (digitalRead(button2) == true);
  delay(20);

  if(pressed1 == true & rotDirection1 ==0){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    rotDirection1 = 1;
    delay(20);
  }
  //If button is pressed - changed rotation direction
  if (pressed1 == false & rotDirection1 == 1){
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    rotDirection1 = 0;
    delay(20);
  }

  if(pressed2 == true & rotDirection2 ==0){
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    rotDirection2 = 1;
    delay(20);
  }
  //If button is pressed - changed rotation direction
  if (pressed2 == false & rotDirection2 == 1){
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    rotDirection2 = 0;
    delay(20);
  }
}
