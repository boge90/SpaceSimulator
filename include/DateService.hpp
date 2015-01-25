#ifndef DATE_SERVICE_H
#define DATE_SERVICE_H

#include <iostream>
#include <string>

class DateService{
	private:
		static int daysInMonth[12];
		static int daysInMonthLeap[12];

	public:
		/**
		* Converts the time representing seconds since "01-01-2000 00:00" to a string 
		**/
		static void getDate(double time, std::string *input);
};

#endif
