//Robot car with Joystick Control

#define enA 9
#define in1 4
#define in2 5
#define enB 10
#define in3 6
#define in4 7

int motorSpeedA = 0;
int motorSpeedB = 0;

void setup(){
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
}

void loop(){
  int xAxis = analogRead(A0);
  int yAxis = analogRead(A1);
  //Y-axis for forward and backward control (0 - 1023, 470 - 550 as center threshold)
  if (yAxis < 470){
    //Motor A backward
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    //Set Motor B backward
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    //Convert the declining Y-axis reading for going backward 
    //from 470 to 0 int 0 to 255 value for the PWM signal for
    //increasing the motor speed.
    motorSpeedA = map(yAxis, 470,0,0,255);
    motorSpeedB = map(yAxis, 470,0,0,255);
  } else if (yAxis > 550){
    //Set the motor forward
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    //Set Motor B forward
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    // Convert the increasing Y-axis readings for
    //going forward from 550 to 1023 into 0 to 255 value 
    //for the PWM signal for increasing the motor speed.
    motorSpeedA = map(yAxis, 550, 1023, 0, 255);
    motorSpeedB = map(yAxis, 550, 1023, 0, 255);
  } else {
    motorSpeedA = 0;
    motorSpeedB = 0;
  }
  //X-axis used for left and right control
  if (xAxis < 470){
    //Convert the declining X-axis readings from 470 to 0
    //into increasing 0 to 255 value.
    int xMapped = map(xAxis, 470,0,0, 255);
    // Move to left - decrease left motor speed, increase right motor speed
    motorSpeedA = motorSpeedA - xMapped;
    motorSpeedB = motorSpeedB + xMapped;
    motorSpeedA = constrain(motorSpeedA,0,255);
    motorSpeedB = constrain(motorSpeedB,0,255);
  }
  if (xAxis > 550){
    // Convert the increasing X-axis readings from 
    // 550 to 1023 into 0 to 255 value
    int xMapped = map(xAxis, 550, 1023,0,255);
    // Move right - decrease right motor speed, increase left motor speed
    motorSpeedA = motorSpeedA + xMapped;
    motorSpeedB = motorSpeedB - xMapped;
    // Confine the range from 0-255
    motorSpeedA = constrain(motorSpeedA,0,255);
    motorSpeedB = constrain(motorSpeedB,0,255);
  }
  /* To prevent buzzing. NOTE: Please test first the motors before
   * enabling this feature.
  if (motorSpeedA < 70){
    motorSpeedA = 0;
  }
  if (motorSpeedB <70){
    motorSpeedB = 0;
  }
  */
  analogWrite(enA, motorSpeedA);
  analogWrite(enB, motorSpeedB);

  
}

