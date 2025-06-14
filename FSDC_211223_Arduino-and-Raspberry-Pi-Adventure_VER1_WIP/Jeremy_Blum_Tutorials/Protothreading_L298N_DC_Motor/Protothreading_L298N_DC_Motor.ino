//Arduino Two Motor Control with Class motorFlasher

class motorFlasher {
    int enA;
    int in1;
    int in2;
    long OnTime;
    long OffTime;
    int motorState;
    unsigned long previousMillis;
    
  public:
    motorFlasher(int temp_enA, int temp_in1, int temp_in2, long temp_OnTime, long temp_OffTime) {

      enA = temp_enA;
      in1 = temp_in1;
      in2 = temp_in2;
      
      pinMode(enA, OUTPUT);
      pinMode(in1, OUTPUT);
      pinMode(in2, OUTPUT);

      OnTime = temp_OnTime;
      OffTime = temp_OffTime;

      motorState;
      previousMillis;

    }
    void initialState(){
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
    }
    
    void Update() {
      unsigned long currentMillis = millis();

      if ((motorState == HIGH) && (currentMillis - previousMillis >= OnTime)) {
        motorState = LOW;  // Turn it off
        previousMillis = currentMillis;  // Remember the time
        analogWrite(enA, 255);  // Update the actual LED
      }
      else if ((motorState == LOW) && (currentMillis - previousMillis >= OffTime)) {
        motorState = HIGH;  // turn it on
        previousMillis = currentMillis;   // Remember the time
        analogWrite(enA, 0);   // Update the actual LED
      }
    }
};

motorFlasher broom1(5, 6, 7, 1000, 500);
motorFlasher broom2(10, 8, 9 , 500, 250);

void setup(){
  broom1.initialState();
  broom2.initialState();
}

void loop(){
  broom1.Update();
  broom2.Update();
}
