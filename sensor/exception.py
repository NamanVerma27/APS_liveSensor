import os, sys


class SensorException(Exception):
    def __init__(self, message: str, error_detail: sys):
        super().__init__(message)
        self.message = message
        self.error_detail = error_detail

    def __str__(self) -> str:
        try:
            _, _, exc_tb = self.error_detail.exc_info()
            if exc_tb is not None:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_no = exc_tb.tb_lineno
                return f"Error Message: {self.message} \nError Detail: {file_name} \nLine Number: {line_no}"
        except Exception:
            pass
        return f"Error Message: {self.message}"