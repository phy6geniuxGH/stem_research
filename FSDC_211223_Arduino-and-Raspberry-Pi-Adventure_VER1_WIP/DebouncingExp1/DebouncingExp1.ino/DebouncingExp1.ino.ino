const int buttonPin = 7;
const int ledPin = 13;

int ledState = HIGH;
int buttonState = HIGH;
int lastButtonState = LOW;

unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;

void setup() {
  pinMode(buttonPin, INPUT);
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, ledState);
  Serial.begin(9600);
}
void loop() {
  
  int reading = digitalRead(buttonPin);
  if(reading != lastButtonState) {
    lastDebounceTime = millis();
    Serial.print("Last Debounce Time: ");
    Serial.println(lastDebounceTime);
  }
  if((millis() - lastDebounceTime) > debounceDelay){
    //Serial.print("Delta Time: ");
    //Serial.println(millis() - lastDebounceTime);
    if (reading != buttonState){
      //Serial.print("Last Button State: ");
      //Serial.println(lastButtonState);
      Serial.print(" Previous Button State: ");
      Serial.print(buttonState);
      buttonState = reading;
      Serial.print(" Current Button State: ");
      Serial.print(buttonState, ",");
      Serial.print(" Previous LED State: ");
      Serial.println(ledState, ",");
      if(buttonState == HIGH){
        ledState = !ledState;
        Serial.print(" Current LED State:");
        Serial.println(ledState, ",");
      }
    }
  }
  digitalWrite(ledPin, ledState);
  lastButtonState = reading;
}
