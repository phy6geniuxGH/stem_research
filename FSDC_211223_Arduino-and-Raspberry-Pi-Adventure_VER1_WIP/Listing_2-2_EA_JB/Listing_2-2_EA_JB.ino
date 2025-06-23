const int LED = 13;
int counter   = 10;  
int max_ms    = 200;
int min_ms    = 10;
void setup() {
  pinMode(LED, OUTPUT);
}

void loop() {
  
  for (int i = min_ms; i <=max_ms; i = i + counter){
    digitalWrite(LED, HIGH);
    delay(i);
    digitalWrite(LED, LOW);
    delay(i);
    if(i >= max_ms || i <= 0){
      counter = -counter;
    }
  }
}
