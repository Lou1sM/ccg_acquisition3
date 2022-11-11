class error_flag:
    def __init__(self):
        self.flag = False

    def set_flag(self):
        self.flag = True

    def reset_flag(self):
        self.flag = False

    def check_flag(self):
        return self.flag


f = error_flag()


def error(string):
    print "ERROR " + string
    f.set_flag()


# abort()

def reset_flag():
    f.reset_flag()


def check_flag():
    return f.check_flag()
