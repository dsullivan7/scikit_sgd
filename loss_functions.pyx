cdef class BaseLossFunction:
    cdef float loss(self, float p, float y):
        pass
    cdef float dloss(self, float p, float y):
        pass

cdef class Hinge(BaseLossFunction):

    cdef void set_threshold(self, float threshold):
        self.threshold = threshold

    cdef float loss(self, float p, float y):
        cdef float z = p * y
        if z <= self.threshold:
            return self.threshold - z
        else:
            return 0

    cdef float dloss(self, float p, float y):
        cdef float z = p * y
        if z <= self.threshold:
            return -y
        else:
            return 0
