#ifndef DRAW_SERVICE_H
#define DRAW_SERVICE_H

#include "../include/Config.hpp"

class DrawService{
	private:
		int width;
		int height;
		unsigned char *pixels;
		
		int CHAR_WIDTH;
		int CHAR_HEIGHT;
		
		// Misc
		size_t debugLevel;
	public:
		DrawService(int width, int height, unsigned char *pixels, Config *config);
		~DrawService();
	
		void fill(unsigned char r, unsigned char g, unsigned char b);
		void fillArea(int x, int y, unsigned char r, unsigned char g, unsigned char b);
		void drawChar(int xc, int yc, char c, unsigned char r, unsigned char g, unsigned char b, int size, bool fill);
		void drawRectangle(int xc, int yc, int width, int heigth, unsigned char r, unsigned char g, unsigned char b, bool fill);
		void drawCircle(int xc, int yc, int radius, unsigned char r, unsigned char g, unsigned char b, bool fill);
		void drawCircleCenter(int xc, int yc, int radius, unsigned char r, unsigned char g, unsigned char b, bool fill);
		void drawLine(int xs, int ys, int xe, int ye, unsigned char r, unsigned char g, unsigned char b);
		void setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);
		
		int widthOf(char c);
		
		unsigned char getRed(int x, int y);
		unsigned char getGreen(int x, int y);
		unsigned char getBlue(int x, int y);
};

#endif
