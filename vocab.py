
class Vocab:

    def __init__(self,w2ind):
        self.w2ind =  w2ind
        self.ind2w = [x for x in w2ind.keys()]
    def __len__(self):
        return len(self.w2ind)
    def map(self,units):
        return [self.w2ind.get(x,UNK_IND) for x in units]

    def unmap(self,idx):
        return [self.ind2w[i] for i in idx]
