const int ledPin = 10; //or pin #13
int ledState = LOW;
unsigned long previousMillis = 0;
const long interval = 1000;

void setup(){
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop(){
  
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval){
    previousMillis = currentMillis;
    Serial.print("Time: ");
    Serial.println(currentMillis);
    Serial.print("Previous Time: ");
    Serial.println(previousMillis);
    Serial.print("Time difference: ");
    Serial.println(currentMillis - previousMillis);
    if (ledState == LOW){
      ledState = HIGH;
    } else {
      ledState = LOW;
    }
    digitalWrite(ledPin, ledState);
  }
}

