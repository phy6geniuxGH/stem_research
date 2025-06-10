const int ledPin = 10; //or pin #13
int ledState = LOW;
unsigned long previousMillis = 0;
const long interval = 10;
int brightness = 0;
int fade = 50;

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
    analogWrite(ledPin, brightness);
    brightness = brightness + fade;
    if (brightness <= 0 || brightness >= 255){
      fade = - fade;
    }
  }
  
}

