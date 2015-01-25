#include "../include/DateService.hpp"

int DateService::daysInMonth[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
int DateService::daysInMonthLeap[12] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

void DateService::getDate(double time, std::string *input){
	long year, day, dayOfYear, month, hour, minute, sec, millis;
	bool isLeapYear;
	
	// Finding year
	year = time / ( 3600 * 24 * 365 );
	
	if( ( year % 4 ) != 0 ){
		isLeapYear = 0;
	}else if( ( year % 100 ) != 0 ){
		isLeapYear = 1;
	}else if( ( year % 400 ) != 0 ){
		isLeapYear = 0;
	}else{
		isLeapYear = 1;
	}
	
	// Finding day
	time -= year * ( 3600 * 24 * 365 );
	day = time / ( 3600 * 24 );
	dayOfYear = day;

	// Finding Month
	int *days;
	if(isLeapYear){
		days = daysInMonthLeap;
	}else{
		days = daysInMonth;
	}
	
	for(month=0; month<12; month++){
		if(day < days[month]){
			break;
		}else{
			day -= days[month];
		}
	}
	
	// Finding hour
	time -= dayOfYear * ( 3600 * 24 );
	hour = time / 3600;
	
	// Finding minute
	time -= hour * 3600;
	minute = time / 60;
	
	// Finding seconds
	time -= minute * 60;
	sec = ( long ) time;
	
	// Millis
	time -= sec;
	millis = ( long ) (time * 1000);
	
	*input += std::to_string(day+1) + ".";
	*input += std::to_string(month+1) + ".";
	*input += std::to_string(year+2000) + "   ";
	*input += std::to_string(hour) + ".";
	*input += std::to_string(minute) + ".";
	*input += std::to_string(sec) + ".";
	*input += std::to_string(millis);
}
