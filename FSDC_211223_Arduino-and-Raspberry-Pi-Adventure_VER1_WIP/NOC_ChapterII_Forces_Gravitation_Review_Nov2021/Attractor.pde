class Attractor {
  float mass;
  PVector location;
  float G;
  
  Attractor(){
    location = new PVector(0, 0);
    mass = 50;
    G = 2;
  }
  
  void display(){
    stroke(0);
    fill(175, 200,255);
    ellipse(location.x, location.y, mass*2, mass*2);
  }
  
  PVector attract(Mover m){
    PVector force = PVector.sub(location, m.location);
    float distance =  force.mag();
    distance = constrain(distance, 50, 60);
    force.normalize();
    
    float strength = (G*mass*m.mass)/(distance * distance);
    
    force.mult(strength);
    
    return force;
  }
}
