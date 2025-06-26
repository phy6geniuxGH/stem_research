const int WLED = 9;         //White LED Anode on pin 9
const int LIGHT = 0;        //Light Sensor on Analog Pin 0
const int MIN_LIGHT = 50;  //Minimum Expected light value
const int MAX_LIGHT = 500;  //Maximum Expected light value
int val = 0;                //Variable to hold the analog reading

void setup() {
  // put your setup code here, to run once:
  pinMode(WLED, OUTPUT); // Set the LED as output
}4

void loop() {
  // put your main code here, to run repeatedly:
  val = analogRead(LIGHT);                          //read the light sensor
  val = map(val, MIN_LIGHT, MAX_LIGHT, 255, 0);     //map the light reading
  val = constrain(val, 0, 255);                     //constrain the light value
  analogWrite(WLED, val);
}
