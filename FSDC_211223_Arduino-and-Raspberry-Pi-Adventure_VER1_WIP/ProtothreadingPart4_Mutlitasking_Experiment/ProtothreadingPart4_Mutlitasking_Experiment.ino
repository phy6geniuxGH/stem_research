#include <Servo.h>

class Flasher {
    int ledPin;
    long OnTime;
    long OffTime;

    int ledState;
    unsigned long previousMillis;

  public:
    Flasher(int temp_ledPin, long temp_OnTime, long temp_OffTime) {
      ledPin = temp_ledPin;
      pinMode(ledPin, OUTPUT);

      OnTime = temp_OnTime;
      OffTime = temp_OffTime;

      ledState;
      previousMillis;
    }
    void Update() {
      unsigned long currentMillis = millis();

      if ((ledState == HIGH) && (currentMillis - previousMillis >= OnTime)) {
        ledState = LOW;  // Turn it off
        previousMillis = currentMillis;  // Remember the time
        digitalWrite(ledPin, ledState);  // Update the actual LED
      }
      else if ((ledState == LOW) && (currentMillis - previousMillis >= OffTime)) {
        ledState = HIGH;  // turn it on
        previousMillis = currentMillis;   // Remember the time
        digitalWrite(ledPin, ledState);   // Update the actual LED
      }
    }
};

class Sweeper {
    Servo servo;
    int pos;
    int increment;
    int updateInterval;
    unsigned long lastUpdate;

  public:
    Sweeper(int temp_updateInterval)
    {
      updateInterval = temp_updateInterval;
      increment = 1;
    }
    void Attach(int pin)
    {
      servo.attach(pin);
    }
    void Detach()
    {
      servo.detach();
    }
    void Update()
    {
      if ((millis() - lastUpdate) > updateInterval)
      {
        lastUpdate = millis();
        pos += increment;
        servo.write(pos);
        Serial.println(pos);
        Serial.println(lastUpdate);
        if ((pos >= 180) || (pos <= 0)) {
          increment = -increment;
        }
      }
    }
};



Flasher led1(12, 100, 400);
Flasher led2(13, 750, 50);
Flasher led3(11, 50, 1050);

Sweeper sweeper1(15);
Sweeper sweeper2(25);

void setup() {
  Serial.begin(9600);
  sweeper1.Attach(9);
  sweeper2.Attach(10);
}

void loop() {
  sweeper1.Update();
  if (digitalRead(7) == HIGH) {
    sweeper2.Update();
    led1.Update();
  }
  led2.Update();
  led3.Update();
}
