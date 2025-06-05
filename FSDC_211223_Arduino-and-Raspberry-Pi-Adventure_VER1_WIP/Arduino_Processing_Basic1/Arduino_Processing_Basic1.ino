int led = 13;

void setup(){
  pinMode(led, OUTPUT);
  Serial.begin(9600);
}

void loop(){
  if(Serial.available()>0){
    char ledState = Serial.read();
    if (ledState == '1'){
      digitalWrite(led, HIGH); 
    }
    if (ledState == '0'){
      digitalWrite(led, LOW);
    }
  }
}

