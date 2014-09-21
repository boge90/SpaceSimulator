#ifndef BMP_H
#define BMP_H

class BMP{
	private:	
		int width;
		int height;
		unsigned char *data;
	public:
		BMP(int width, int height, unsigned char *data);
		~BMP(void);
		
		/**
		* returns a pointer to the data of the image
		**/
		unsigned char* getData(void);
		
		/**
		* returns the height of the image
		**/
		int getHeight(void);
		
		/**
		* returns the width of the image
		**/
		int getWidth(void);
};

#endif
