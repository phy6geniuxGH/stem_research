Mover[] movers = new Mover[100];   //Create 100 Mover Objects
Liquid liquid;
Attractor attractor;

float particle_mass;
float init_x, init_y;

boolean y_grav = false;
boolean wind_force = false;
boolean friction_force = false;
boolean fluid_res = false;
boolean grav_attract = true;

void setup(){
  size(700,700);
  init_x        = width/2;
  init_y        = height/2;
  particle_mass = 30;
  for (int i = 0; i < movers.length; i++){
    movers[i] = new Mover(random(particle_mass), random(-init_x,init_x), random(-init_y,0));
    
  }
  //Setup the Liquid
  liquid = new Liquid(0,0, width/6, height/6, 0.1);
  attractor = new Attractor();
}

void draw(){
  background(0);
  //Set Coordinate Settings
  translate(width/2, height/2);
  scale(1, 1); 
  rotate(0);
  //Show liquid
  if (fluid_res){
    liquid.display();
  }
  
  if (grav_attract){
    attractor.display();
  }
  //Add the forces
  for (int i = 0; i < movers.length; i++){
    //Fluid Drag
    if(fluid_res){
      if (movers[i].isInside(liquid)){
        movers[i].drag(liquid, movers[i]);
      }
    }
    
    if(wind_force){
      PVector wind = new PVector(0,0);
      if (mousePressed && (mouseButton == LEFT)) {
        wind = new PVector(0.01,0);
        wind.mult(-1);
      } else if (mousePressed && (mouseButton == CENTER)) {
        wind.mult(0);
      } else if (mousePressed && (mouseButton == RIGHT)) {
        wind = new PVector(0.01,0);
      }
      movers[i].applyForce(wind);
    }
    
    //Gravity Scaled by Mass
    if(y_grav){
      float m = movers[i].mass;
      PVector gravity  = new PVector(0, 0.1*m);
      movers[i].applyForce(gravity);
    }
    //Add Gravitational Attraction
    if(grav_attract){
      PVector gravForce = attractor.attract(movers[i]);
      movers[i].applyForce(gravForce);
    }
    
    //Update the calculations
    movers[i].update();
    movers[i].display();
    movers[i].checkEdges(movers[i]);
  }
}
