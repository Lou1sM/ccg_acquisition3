class CCGLearnerError(Exception):
    """A base class for CCGLearnerError exceptions."""
    def __init__(self, msg=''):
        self.msg = msg

    def __repr__(self):
        raise NotImplementedError

class SemCatError(CCGLearnerError):
    """A base class for CCGLearnerError exceptions."""

    def __str__(self):
        return 'SemCat Error' if self.msg == '' else f'SemCat Error: {self.msg}'

class SynCatError(CCGLearnerError):
    """A base class for CCGLearnerError exceptions."""

    def __str__(self):
        return 'SynCat Error' if self.msg == '' else f'SynCat Error: {self.msg}'

class ZeroProbError(CCGLearnerError):
    """A base class for CCGLearnerError exceptions."""

    def __str__(self):
        return 'ZeroProb Error' if self.msg == '' else f'ZeroProb Error: {self.msg}'

class RootSemCatError(CCGLearnerError):
    """A base class for CCGLearnerError exceptions."""

    def __str__(self):
        return f'Unable to infer root semcat for: {self.msg}'

