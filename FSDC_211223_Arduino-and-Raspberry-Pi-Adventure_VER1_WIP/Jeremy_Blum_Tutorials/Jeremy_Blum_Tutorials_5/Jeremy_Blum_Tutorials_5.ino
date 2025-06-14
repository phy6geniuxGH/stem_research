//Voltage Divider, Analog Input Reading through Serial port.

int sensePin = 0;

void setup() {
  analogReference(DEFAULT);//isn't necessary
  Serial.begin(9600);

}

void loop() {
  Serial.println(analogRead(sensePin));
  delay(500);
}

//PR1 first then 10k ohm resistor - from 850 down to 160~
//10k ohm first then PR1 - from 160~ to 850
//reversing the position will change the output analog signal.
