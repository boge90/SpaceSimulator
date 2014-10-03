#ifndef TEXT_VIEW_H
#define TEXT_VIEW_H

#include "../include/View.hpp"
#include "../include/Config.hpp"
#include <string>

class TextView: public View{
	protected:
		std::string text;
		std::string previousText;
		
		int leftPadding;
		int topPadding;
		int charPadding;
		bool repaintFlag;
		
		/**
		* Draws the string
		**/
		void drawText(std::string text, DrawService *drawService, unsigned char r, unsigned char g, unsigned char b);
	public:
		/**
		* Creates a TextView object with the specified text
		**/
		TextView(std::string text, Config *config);
		
		/**
		* Finalizes the TextView object
		**/
		~TextView(void);
		
		/**
		* Called when the view should be drawn
		**/
		void draw(DrawService *drawService);
		
		/**
		* Updates the text string
		**/
		void setText(std::string text);
		
		/**
		* Returns the pointer to the text string
		**/
		std::string* getText();
		
		/**
		*
		**/
		void repaint(void);
};

#endif
