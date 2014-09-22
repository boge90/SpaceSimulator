#ifndef TEXT_VIEW_H
#define TEXT_VIEW_H

#include "../include/View.hpp"
#include <string>

class TextView: public View{
	protected:
		std::string text;
		int leftPadding;
		int topPadding;
		int charPadding;
		
	public:
		/**
		*
		**/
		TextView(std::string text);
		
		/**
		*
		**/
		~TextView(void);
		
		/**
		*
		**/
		void draw(DrawService *drawService);
};

#endif
