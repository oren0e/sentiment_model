# mycode/wsgi.py

from mycode.app import app
from mycode.model import text_process

if __name__ == '__main__':
    from model import *
    app.run()


