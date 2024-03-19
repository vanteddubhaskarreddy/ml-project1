import sys
import logging
import logger

# sys module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
# sys.exc_info() function returns a tuple of three values that give information about the exception that is currently being handled.
# logging module defines functions and classes which implement a flexible event logging system for applications and libraries.
# logger module is a custom module which is used to log the error message in the log file.

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = 'Error in python script :[{0}] \t on line number : [{1}] \t with error message : [{2}]'.format(file_name,exc_tb.tb_lineno,str(error) )
    return error_message
    # This function will print the error message with the file name and line number where the error occured.
    #_,_,exc_tb = error_detail.exc_info() will get the file name and line number where the error occured.
    #The first two arguments are the type of exception and the value of the exception. The third argument is the traceback.
    #exc_tb.tb_frame.f_code.co_filename here order is traceback frame,frame code,code filename will get the file name where the error occured.
    #the parameter of function error_message_detail is error and error_detail:sys. error is the error message and error_detail:sys is the traceback of the error.

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail = error_detail)
    
    def __str__(self):
        return self.error_message

    #This class will create a custom exception with the error message and error detail inherited from the Exception class.
    #The __init__ function will initialize the error message and error detail.
    #The super().__init__(error_message) will call the parent class constructor and initialize the error message.
    #super().__init__(error_message) This line calls the constructor of the base class (Exception). This is necessary because CustomException is a subclass of Exception, and we want to make sure that any initialization that Exception does is also done for CustomException. The error_message is passed to the Exception constructor, which uses it to set the error message for the exception.
    #The error_message_detail function will return the error message with the file name and line number where the error occured.
    #The __str__ function will return the error message.

if __name__ == '__main__':
    try:
        a = 1/0
    except Exception as e:
        logger.logging.info('Divide by zero error')
        raise CustomException(e,sys)