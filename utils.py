import logging


def log(
    before_msg="",
    after_msg="",
    logging_func=logging.info,
    debug=False,
):
    """Wrapper that gives a function a start and end logging message.

    Args:
        logging_func (func): Python logging function to be used
        before_msg (str): Message passed to logging before the primary function is called.
        after_msg (str): Message passed to logging after the primary function is called.
        debug (bool): If true, some additional function details will be prepended to the before_msg.
    """
    def decorate(func):
        """ Decorator """
        def call(*args, **kwargs):
            """ Actual wrapping """
            debug_msg: str = ""
            if debug:
                debug_msg = "{{{file_name}:{line_no}:{func_name}}} - ".format(
                        file_name=func.__code__.co_filename,
                        line_no=func.__code__.co_firstlineno,
                        func_name=func.__name__
                )
            if before_msg != "" or debug_msg != "":
                logging_func(debug_msg + before_msg)
            result: func = func(*args, **kwargs)
            if after_msg != "":
                logging_func(after_msg)
            return result
        return call
    return decorate
