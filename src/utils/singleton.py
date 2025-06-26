import logging

class Singleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            logging.info(' Singleton: initial instance. Intantiating a new one.')
            cls.__instance = super().__call__(*args, **kwargs)
        return cls.__instance
