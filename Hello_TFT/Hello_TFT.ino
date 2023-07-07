#include <SPI.h>
#include <TFT.h>            // Arduino TFT library

#define cs   10
#define dc   8
#define rst  9

TFT screen = TFT(cs, dc, rst);

void setup() {
  // initialize the screen
  screen.begin();
  // make the background black
  screen.background(0, 109, 112);
  // set the stroke color to white
  screen.stroke(0,255,0);
  // set the fill color to grey
  screen.setTextSize(2);
  screen.text("Hi TFT!", 30, 30);
}

void loop() {

}
