class KFold():
    def __init__(self, maxk):
        self.maxk = maxk
        self.current_k = 0

    def increaseK():
        self.current_k = (self.current_k+1)%self.maxk

    def getCurrentK():
        return self.current_k
