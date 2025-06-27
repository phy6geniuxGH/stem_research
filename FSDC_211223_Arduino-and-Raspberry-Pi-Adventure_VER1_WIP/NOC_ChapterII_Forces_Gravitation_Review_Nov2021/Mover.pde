class Mover{
  PVector location;        //Create a location vector  
  PVector velocity;        //Create a velocity vector
  PVector acceleration;    //Create an acceleration vector
  
  float mass;              //add mass
  float radius;
  
  //Define a Mover object
  Mover(float temp_m, float temp_x, float temp_y){
    location       = new PVector(temp_x, temp_y);
    velocity       = new PVector(0,0);
    acceleration   = new PVector(0,0);
    mass     = temp_m;
    radius   = mass/2;
  }
  
  void applyForce(PVector force){    //Create a function that takes a force vector
    //get a copy of force first, not to be overwritten when changes were applied
    //PVector f = force.copy();
    //Solve F/m (Newton's Second Law)
    //f.div(mass);
    //Static Version
    PVector f = PVector.div(force, mass);
    //Force Accumulation: (Note that the f now is the acceleration
    acceleration.add(f);
  }
  
  void update(){       //Create a function that updates the vectors
    velocity.add(acceleration);
    location.add(velocity);
    acceleration.mult(0);   //reset the acceleration value
  }
  
  void display(){
    noStroke();
    fill(255);
    ellipse(location.x, location.y, radius, radius);   
  }
  
  void checkEdges(Mover m){    //Create a function that tells the limits of the simulation space
    if (location.x + m.radius/2> width/2){
      location.x = width/2 - m.radius/2;
      velocity.x *= -1;
    } else if (location.x+m.radius/2 < -width/2){
      location.x = -width/2 + m.radius/2;
      velocity.x *= -1;
    }
    
    if (location.y+m.radius/2> height/2){
      location.y = height/2 - m.radius/2;
      velocity.y *= -1;
    } else if (location.y+m.radius/2 < -height/2){
      location.y = -height/2 + m.radius/2;
      velocity.y *= -1;
    }
  
  }
  
  //Define if the mover is inside the liquid
  boolean isInside(Liquid l){
    if (location.x > l.x && location.x < l.x+l.w && location.y > l.y && location.y <l.y +l.h){
      return true;
    } else {
      return false;
    }
  }
  
  //Create the drag function within the mover class
  void drag(Liquid l, Mover m){
    float speed = velocity.mag();
    float dragMagnitude = l.c*speed*speed;
    PVector drag = velocity.copy();
    drag.mult(-1);
    drag.normalize();
    drag.mult(dragMagnitude*m.radius);
    applyForce(drag);
  }
}
